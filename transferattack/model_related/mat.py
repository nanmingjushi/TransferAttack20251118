"""
  @Project ：bfa.py 
  @File ：mat.py
  @IDE ：PyCharm 
  @Author ：26354
  @Date ：2025/12/11 18:42 
"""


# file: transferattack/model_related/mat.py

from functools import partial
import torch

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class MAT(MIFGSM):
    """
    Multi-space Alignment Transfer Attack (MAT)

    在标准梯度攻击框架（如 MIFGSM）基础上，从三个空间同时对齐：
      - 特征空间：中层随机噪声注入（Mid-layer Noise Injection）
      - 注意力空间：动态注意力温度（Dynamic Attention Temperature）
      - 梯度空间：软 token 梯度整形（Soft Token Gradient Shaping）

    为了与当前最优实验结果严格对应，本实现中所有关键超参数写死：
        中层噪声：beta = 0.01，作用层为第 3–6 层 block
        动态温度：tau_min = 1.0，tau_max = 1.3（随层号线性递增）
        梯度整形：alpha_mid = 1.0, alpha_high = 2.0,
                  s_mid = 0.8, s_high = 0.5

    其它攻击超参数（epsilon, alpha, epoch, decay 等）
    仍沿用 MIFGSM 框架的原有配置。
    """

    def __init__(self, **kwargs):
        self.model_name = kwargs["model_name"]
        kwargs["attack"] = "MAT"

        # 使用固定配置，不再从 kwargs 读取策略相关超参数
        super().__init__(**kwargs)

        # unwrap model（原框架中 model 通常是一个 tuple/list）
        self.model = self.model[1]

        # 注册三个空间上的 hook
        self._register_grad_hooks()                  # 梯度空间：soft shaping
        self._register_noise_hooks()                 # 特征空间：中层噪声
        self._register_attention_temperature_hooks() # 注意力空间：动态温度

        # 重新 wrap 回去
        self.model = wrap_model(self.model.eval().cuda())

    # ------------------------------------------------------------------
    #  Forward: 中层随机噪声注入（特征空间）
    # ------------------------------------------------------------------
    def _register_noise_hooks(self):
        """
        在 ViT 的中间几层 blocks 上注册 forward_pre_hook，
        在 block 输入 x（token embedding）上添加随机噪声。

        固定配置：
            噪声强度 beta = 0.01
            作用层索引：3 ~ 6（含）
        """

        NOISE_BETA = 0.01
        NOISE_LAYER_START = 3
        NOISE_LAYER_END = 6

        def midlayer_noise_hook(module, inputs, layer_id, beta):
            """
            inputs: tuple, 通常只有一个元素 x
            x: [B, N, C] 或类似形状
            """
            if len(inputs) == 0:
                return inputs

            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs

            # 使用 token L2 范数均值作为尺度，保证噪声相对强度稳定
            with torch.no_grad():
                feat_norm = x.detach().pow(2).sum(dim=-1).sqrt()  # [B, N]
                mean_norm = feat_norm.mean()
                sigma = beta * mean_norm
                noise = torch.randn_like(x) * sigma

            x_noisy = x + noise
            new_inputs = (x_noisy,) + tuple(inputs[1:])
            return new_inputs

        # 目前仅处理有 .blocks 的 ViT/DeiT 模型
        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            num_blocks = len(self.model.blocks)
            s = max(0, NOISE_LAYER_START)
            e = min(num_blocks - 1, NOISE_LAYER_END)
            for i in range(s, e + 1):
                hook_fn = partial(
                    midlayer_noise_hook,
                    layer_id=i,
                    beta=NOISE_BETA,
                )
                self.model.blocks[i].register_forward_pre_hook(hook_fn)

    # ------------------------------------------------------------------
    #  Forward: Dynamic Attention Temperature（注意力空间）
    # ------------------------------------------------------------------
    def _register_attention_temperature_hooks(self):
        """
        在注意力模块的 attn_drop 上注册 forward_pre_hook，
        对 softmax 后的 attention map 做温度重标定：

            attn' = softmax( log(attn + eps) / tau_l )

        固定配置：
            tau_min = 1.0
            tau_max = 1.3
            随 layer_id 在 [0, L-1] 上线性插值。
        """

        ENABLE_DAT = True
        TAU_MIN = 1.0
        TAU_MAX = 1.3
        EPS = 1e-6

        if not ENABLE_DAT:
            return

        # 目前仅处理 ViT / DeiT 源模型
        if self.model_name not in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            return

        num_blocks = len(self.model.blocks)

        def attn_temperature_hook(
            module,
            inputs,
            layer_id,
            num_layers,
            tau_min,
            tau_max,
            eps,
        ):
            """
            inputs[0] 是进入 attn_drop 的 attention map（通常为 [B, heads, N, N]）。
            """
            if len(inputs) == 0:
                return inputs

            attn = inputs[0]
            if not torch.is_tensor(attn):
                return inputs

            # 计算本层温度 tau_l：从 tau_min 线性插值到 tau_max
            if num_layers > 1:
                t = float(layer_id) / float(num_layers - 1)
            else:
                t = 0.0
            tau = tau_min + (tau_max - tau_min) * t

            # 当 tau 非常接近 1.0 时，可视为无变化，直接返回
            if abs(tau - 1.0) < 1e-6:
                return inputs

            # 使用 log(attn) / tau 再做 softmax，实现温度重标定
            attn_log = (attn + eps).log()
            attn_scaled = (attn_log / tau).softmax(dim=-1)

            new_inputs = (attn_scaled,) + tuple(inputs[1:])
            return new_inputs

        for i in range(num_blocks):
            hook_fn = partial(
                attn_temperature_hook,
                layer_id=i,
                num_layers=num_blocks,
                tau_min=TAU_MIN,
                tau_max=TAU_MAX,
                eps=EPS,
            )
            self.model.blocks[i].attn.attn_drop.register_forward_pre_hook(hook_fn)

    # ------------------------------------------------------------------
    #  Backward: Soft Token Gradient Shaping（梯度空间）
    # ------------------------------------------------------------------
    def _register_grad_hooks(self):
        """
        Soft Token Gradient Shaping（token-wise）：

        统一对 QKV / Attention / MLP 的输入梯度做 token 级整形。

        固定配置：
            alpha_mid = 1.0
            alpha_high = 2.0
            s_mid = 0.8
            s_high = 0.5
        """

        ALPHA_MID = 1.0
        ALPHA_HIGH = 2.0
        S_MID = 0.8
        S_HIGH = 0.5
        EPS = 1e-6

        def soft_token_grad_shaping(grad):
            """
            支持常见梯度形状：
                - [B, N, C]
                - [N, C]（视作 B=1）
                - [B, C, H, W]（H,W 展平为 tokens）
            """
            if grad is None:
                return grad

            dim = grad.dim()
            if dim < 2:
                return grad  # 标量等，直接跳过

            layout = None

            if dim == 2:
                # [N, C] -> [1, N, C]
                g = grad.unsqueeze(0)
                B, T, C = g.shape
                layout = ("2d", None, None)

            elif dim == 3:
                # [B, N, C]
                B, T, C = grad.shape
                g = grad
                layout = ("3d", None, None)

            elif dim == 4:
                # 解释为 [B, C, H, W]，将 H,W 展平为 tokens
                B, C, H, W = grad.shape
                T = H * W
                g = grad.view(B, C, T).permute(0, 2, 1)  # [B, T, C]
                layout = ("4d", H, W)

            else:
                return grad

            # g: [B, T, C]
            token_norm = g.norm(p=2, dim=-1)  # [B, T]

            mean = token_norm.mean()
            std = token_norm.std()

            if std.item() < 1e-6:
                return grad

            z = (token_norm - mean) / (std + EPS)  # [B, T]

            scales = torch.ones_like(token_norm)
            high_mask = z > ALPHA_HIGH
            mid_mask = (z > ALPHA_MID) & (~high_mask)

            scales[high_mask] = S_HIGH
            scales[mid_mask] = S_MID

            scales = scales.unsqueeze(-1)  # [B, T, 1]
            g_shaped = g * scales          # [B, T, C]

            # 还原至原始形状
            if layout[0] == "2d":
                return g_shaped.squeeze(0)               # [N, C]
            elif layout[0] == "3d":
                return g_shaped                          # [B, N, C]
            elif layout[0] == "4d":
                H, W = layout[1], layout[2]
                return g_shaped.permute(0, 2, 1).view(B, C, H, W)
            else:
                return grad

        # ---------- hook 函数：分别作用在 attn / qkv / mlp 模块 ----------

        def attn_hook(module, grad_in, grad_out, gamma):
            # grad_in[0]: [B, heads, N, N]
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad = soft_token_grad_shaping(out_grad)
            return (out_grad,)

        def attn_cait_hook(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad = soft_token_grad_shaping(out_grad)
            return (out_grad,)

        def q_hook(module, grad_in, grad_out, gamma):
            # CaiT 的 Q 只用 class token，这里仍然将其梯度置零，
            # 以避免 class-attention 对整体方向产生过强主导。
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad[:] = 0.0
            return (out_grad, grad_in[1], grad_in[2])

        def v_hook(module, grad_in, grad_out, gamma):
            is_high_pytorch = False
            gi0 = grad_in[0]
            if len(gi0.shape) == 2:
                gi0 = gi0.unsqueeze(0)  # [N, C] -> [1, N, C]
                is_high_pytorch = True

            mask = torch.ones_like(gi0) * gamma
            out_grad = mask * gi0[:]
            out_grad = soft_token_grad_shaping(out_grad)

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)

            return_dics = (out_grad,)
            for i in range(1, len(grad_in)):
                return_dics = return_dics + (grad_in[i],)
            return return_dics

        def mlp_hook(module, grad_in, grad_out, gamma):
            is_high_pytorch = False
            gi0 = grad_in[0]
            if len(gi0.shape) == 2:
                gi0 = gi0.unsqueeze(0)
                is_high_pytorch = True

            mask = torch.ones_like(gi0) * gamma
            out_grad = mask * gi0[:]
            out_grad = soft_token_grad_shaping(out_grad)

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)

            return_dics = (out_grad,)
            for i in range(1, len(grad_in)):
                return_dics = return_dics + (grad_in[i],)
            return return_dics

        # ---------- hook 注册（对不同结构挂到对应模块上） ----------

        attn_mat_hook = partial(attn_hook, gamma=0.25)
        attn_cait_mat_hook = partial(attn_cait_hook, gamma=0.25)
        v_mat_hook = partial(v_hook, gamma=0.75)
        q_mat_hook = partial(q_hook, gamma=0.75)
        mlp_mat_hook = partial(mlp_hook, gamma=0.5)

        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_mat_hook)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_mat_hook)
                self.model.blocks[i].mlp.register_backward_hook(mlp_mat_hook)

        elif self.model_name == "pit_b_224":
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                else:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                blk = self.model.transformers[transformer_ind].blocks[used_block_ind]
                blk.attn.attn_drop.register_backward_hook(attn_mat_hook)
                blk.attn.qkv.register_backward_hook(v_mat_hook)
                blk.mlp.register_backward_hook(mlp_mat_hook)

        elif self.model_name == "cait_s24_224":
            for block_ind in range(26):
                if block_ind < 24:
                    blk = self.model.blocks[block_ind]
                    blk.attn.attn_drop.register_backward_hook(attn_mat_hook)
                    blk.attn.qkv.register_backward_hook(v_mat_hook)
                    blk.mlp.register_backward_hook(mlp_mat_hook)
                else:
                    blk = self.model.blocks_token_only[block_ind - 24]
                    blk.attn.attn_drop.register_backward_hook(attn_cait_mat_hook)
                    blk.attn.q.register_backward_hook(q_mat_hook)
                    blk.attn.k.register_backward_hook(v_mat_hook)
                    blk.attn.v.register_backward_hook(v_mat_hook)
                    blk.mlp.register_backward_hook(mlp_mat_hook)

        elif self.model_name == "visformer_small":
            for block_ind in range(8):
                if block_ind < 4:
                    blk = self.model.stage2[block_ind]
                else:
                    blk = self.model.stage3[block_ind - 4]
                blk.attn.attn_drop.register_backward_hook(attn_mat_hook)
                blk.attn.qkv.register_backward_hook(v_mat_hook)
                blk.mlp.register_backward_hook(mlp_mat_hook)
