"""
  @Project ：TransferAttack20251118 
  @File ：ffnfda.py
  @IDE ：PyCharm 
  @Author ：26354
  @Date ：2025/12/4 16:01 
"""


import torch
import torch.nn as nn

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class FFNFDA(MIFGSM):
    """
    FFN-FDA: Feed-Forward Network Feature Disruption Attack
    基于 ViT FFN 特征空间的迁移攻击（借鉴 FIA 思想）。

    核心思想：
        1) 在若干个 ViT block 的 FFN 中间层 (mlp.fc1) 上估计特征重要性（通过多次随机drop+反向聚合梯度）。
        2) 在迭代攻击中，优先破坏这些重要的 FFN 特征子空间，以提升迁移至其他 ViT 变体的攻击成功率。

    Arguments:
        model_name (str): surrogate 模型名（当前主要支持 ViT，如 vit_base_patch16_224）
        epsilon (float): L_inf 扰动预算
        alpha (float): 每步步长
        epoch (int): 迭代步数
        decay (float): 动量衰减系数（继承 MIFGSM，但这里我们不用动量，而是沿用 update_delta 接口）
        num_ens (int): 重要性聚合时的随机增强次数
        ffn_layers (str): 选取的 FFN 层索引，逗号分隔，如 "2,6,10"
        patch_only (bool): 是否只使用 patch token（丢掉 CLS）
        drop_rate (float): FIA 风格的随机 drop 概率
        targeted (bool): 是否为目标攻击
        norm (str): "linfty" 或 "l2"
        loss (str): 这里用在基类里（我们特征损失自己写）
        device (torch.device): 设备
        attack (str): 攻击名称（用于日志）

    Example:
        python main.py --input_dir ... --output_dir ... \
            --attack ffnfda --model vit_base_patch16_224 --batchsize 1
    """

    def __init__(self,
                 model_name='vit_base_patch16_224',
                 epsilon=16/255,
                 alpha=1.6/255,
                 epoch=10,
                 decay=1.,
                 num_ens=30,
                 ffn_layers='2,6,10',
                 patch_only=True,
                 drop_rate=0.3,
                 targeted=False,
                 random_start=False,
                 norm='linfty',
                 loss='crossentropy',
                 device=None,
                 attack='ffnfda',
                 **kwargs):

        super().__init__(model_name, epsilon, alpha, epoch, decay,
                         targeted, random_start, norm, loss, device, attack)

        # 记录配置
        self.model_name = model_name
        self.num_ens = num_ens
        self.patch_only = patch_only
        self.drop_rate = drop_rate

        # 真正的背骨模型（用于挂 hook），self.model 是 wrap_model(...)
        self.backbone = self.model[1]

        # 解析要用哪些 FFN 层（block index）
        self.ffn_layer_ids = [int(x) for x in ffn_layers.split(',') if x != '']

        # 只先支持标准 ViT 结构，防止乱挂
        if not (model_name.startswith('vit') or 'deit' in model_name):
            print(f"[FFNFDA] WARNING: {model_name} 不是标准 ViT，当前版本只对 ViT 系结构测试过，请谨慎使用。")

        # 用于存储前向特征和反向梯度（按 layer id 记录）
        self.mid_outputs = {idx: None for idx in self.ffn_layer_ids}
        self.mid_grads = {idx: None for idx in self.ffn_layer_ids}
        self.agg_importance = {idx: None for idx in self.ffn_layer_ids}

        # 保存 hook handle，方便最后 remove
        self.fwd_hooks = []
        self.bwd_hooks = []

        # 注册 FFN 层的 forward/backward hook（只注册一次，整个攻击过程中复用）
        self._register_ffn_hooks()

    # ----------------------------------------------------------------------
    # Hook 注册
    # ----------------------------------------------------------------------
    def _register_ffn_hooks(self):
        """
        在指定的 FFN 层 (mlp.fc1) 上注册 forward / backward hooks。
        """
        for idx in self.ffn_layer_ids:
            try:
                # timm 的 ViT: blocks[idx].mlp.fc1
                layer = self.backbone.blocks[idx].mlp.fc1
            except Exception as e:
                print(f"[FFNFDA] ERROR: 无法在 {self.model_name} 上找到 blocks[{idx}].mlp.fc1: {e}")
                continue

            # forward hook: 记录 FFN 中间激活
            def fwd_hook_closure(layer_id):
                def fwd_hook(module, input, output):
                    # output: (B, T, d_ff)
                    self.mid_outputs[layer_id] = output
                return fwd_hook

            # backward hook: 记录对 FFN 中间激活的梯度
            def bwd_hook_closure(layer_id):
                def bwd_hook(module, grad_input, grad_output):
                    # grad_output[0]: (B, T, d_ff)
                    self.mid_grads[layer_id] = grad_output[0]
                return bwd_hook

            self.fwd_hooks.append(
                layer.register_forward_hook(fwd_hook_closure(idx))
            )
            # 用 full_backward_hook 更稳定一些（参考 FIA）
            self.bwd_hooks.append(
                layer.register_full_backward_hook(bwd_hook_closure(idx))
            )

    def _remove_hooks(self):
        for h in self.fwd_hooks:
            h.remove()
        for h in self.bwd_hooks:
            h.remove()
        self.fwd_hooks = []
        self.bwd_hooks = []

    # ----------------------------------------------------------------------
    # 随机 drop（借鉴 FIA）
    # ----------------------------------------------------------------------
    def _drop(self, data):
        """
        类似 FIA 的随机 Drop：对输入乘一个 Bernoulli Mask。
        """
        x_drop = data.clone().detach().to(self.device)
        x_drop.requires_grad = True
        mask = torch.bernoulli(torch.ones_like(x_drop) * (1 - self.drop_rate)).to(self.device)
        x_drop = x_drop * mask
        return x_drop

    # ----------------------------------------------------------------------
    # 估计 FFN 特征重要性（类似 FIA 的聚合梯度）
    # ----------------------------------------------------------------------
    def _estimate_ffn_importance(self, data, label):
        """
        在 clean image 上，对指定的 FFN 层估计特征重要性：
            通过 num_ens 次随机 drop + backward，在 FFN 中间激活上聚合梯度。
        结果保存在 self.agg_importance[layer_id] 中，形状与 mid_outputs 相同。
        """
        # 重置聚合器
        for idx in self.ffn_layer_ids:
            self.agg_importance[idx] = None

        B = data.shape[0]

        # 多次随机增强 / dropout
        for _ in range(self.num_ens):
            x_drop = self._drop(data)
            logits = self.model(x_drop)          # forward，触发 forward hook，填充 mid_outputs
            probs = torch.softmax(logits, dim=1)

            # 这里参考 FIA：最大化当前真类的概率
            # untargeted: 提升真类 prob，然后在构造特征 loss 时相当于往反方向走
            loss = 0.
            for b in range(B):
                loss += probs[b, label[b]]

            self.model.zero_grad()
            loss.backward()

            # backward 触发，mid_grads 被填充
            for idx in self.ffn_layer_ids:
                g = self.mid_grads[idx]
                if g is None:
                    continue
                g = g.detach()
                if self.agg_importance[idx] is None:
                    self.agg_importance[idx] = g.clone()
                else:
                    self.agg_importance[idx] += g

        # 对每个 layer 的重要性做 L2 归一化（按每张图的整体特征向量）
        for idx in self.ffn_layer_ids:
            g = self.agg_importance[idx]
            if g is None:
                continue
            # g: (B, T, d_ff)
            flat = g.view(B, -1)
            norm = flat.norm(p=2, dim=1, keepdim=True) + 1e-8
            g = g / norm.view(B, 1, 1)
            self.agg_importance[idx] = g

        # 清一下梯度缓存
        self.model.zero_grad()

    # ----------------------------------------------------------------------
    # 主攻击过程
    # ----------------------------------------------------------------------
    def forward(self, data, label, **kwargs):
        """
        整体攻击流程：
            1) 对 clean data 估计 FFN 特征重要性 self.agg_importance。
            2) 用特征损失在像素空间迭代更新 delta。
        """
        # 处理 targeted 场景（参考其他攻击）
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # 初始化扰动
        delta = self.init_delta(data)

        # 第一步：在 clean data 上估计 FFN 特征重要性
        self._estimate_ffn_importance(data, label)

        # 第二步：迭代攻击（只用 FFN 特征损失，不额外加 CE，先最大化迁移性）
        for _ in range(self.epoch):
            # 让模型跑一遍前向（带当前 delta），填充 mid_outputs
            logits = self.get_logits(self.transform(data + delta))

            # 构造 FFN 特征损失
            feature_loss = 0.
            for idx in self.ffn_layer_ids:
                feat = self.mid_outputs[idx]            # (B, T, d_ff)
                imp = self.agg_importance[idx]          # (B, T, d_ff)
                if feat is None or imp is None:
                    continue

                # 只使用 patch token（丢 CLS），提升跨模型迁移性
                if self.patch_only:
                    feat = feat[:, 1:, :]
                    imp = imp[:, 1:, :]

                # 内积：让当前特征在“重要方向”上发生大变化
                feature_loss = feature_loss + (feat * imp).sum()

            # 计算对 delta 的梯度
            self.model.zero_grad()
            grad = torch.autograd.grad(feature_loss, delta,
                                       retain_graph=False,
                                       create_graph=False)[0]

            # 按 FIA 方式，用 -grad 更新（即破坏这些重要特征）
            delta = self.update_delta(delta, data, -grad, self.alpha)

        return delta.detach()



