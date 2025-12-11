"""
  @Project ：bfa.py 
  @File ：vit_cam_demo.py
  @IDE ：PyCharm 
  @Author ：26354
  @Date ：2025/12/11 18:44 
"""

# vit_cam_demo.py
# -*- coding: utf-8 -*-
"""
使用 ViTGradCAM 可视化 ViT 及其变体在 clean / adv 图像上的注意力区域。

步骤：
  1) 加载预训练 ViT-B/16 模型（也可以换成你代码里的 ViT）
  2) 读取本地 clean / adv 图像
  3) 使用 ViTGradCAM 计算 CAM
  4) 将 CAM 叠加到原图上并保存

运行方式：
  python vit_cam_demo.py
"""

import os
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn

try:
    import timm  # 用来加载 ViT-B/16 预训练模型
except ImportError:
    timm = None
    print("Warning: timm 未安装，如果你想用预训练 ViT，请先: pip install timm")


# ------------------------ 一、ViTGradCAM 实现 ------------------------


class ViTGradCAM(object):
    """
    面向 ViT 及其变体的 Grad-CAM（同时兼容卷积特征）。
    - 对 ViT: 处理 [B, N, C] 的 token 特征，去掉 cls token 后 reshape 成 H×W。
    - 对 CNN: 处理 [B, C, H, W] 的卷积特征，退化成普通 Grad-CAM。

    用法：
        cam = ViTGradCAM(model, layer_name='blocks.11.norm1', img_size=224)
        heatmap = cam(inputs, index=None)  # inputs: [1,3,224,224]
    """

    def __init__(self, net, layer_name, img_size=224):
        """
        Args:
            net:       要可视化的模型 (nn.Module)
            layer_name: 要 hook 的层名（用 net.named_modules() 查）
                        ViT: 比如 'blocks.11.norm1'、'blocks.11.attn'
                        Swin / Visformer: 可以 hook 到某个 conv/attention 模块
            img_size:  输出 CAM 的分辨率（通常等于输入图像分辨率，如 224）
        """
        self.net = net
        self.layer_name = layer_name
        self.img_size = img_size

        self.feature = None
        self.gradient = None

        self.net.eval()
        self.handlers = []
        self._register_hook()

    # ---------------- hook 部分 ----------------
    def _forward_hook(self, module, inp, out):
        # 保存 forward 输出特征
        self.feature = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        # 保存输出的梯度：grad_out[0] 对应 self.feature 的梯度
        self.gradient = grad_out[0].detach()

    def _register_hook(self):
        found = False
        for name, module in self.net.named_modules():
            if name == self.layer_name:
                found = True
                self.handlers.append(
                    module.register_forward_hook(self._forward_hook)
                )
                # 新版 PyTorch 推荐 full_backward_hook
                try:
                    self.handlers.append(
                        module.register_full_backward_hook(self._backward_hook)
                    )
                except AttributeError:
                    self.handlers.append(
                        module.register_backward_hook(self._backward_hook)
                    )
        if not found:
            print(f"[ViTGradCAM] Warning: layer {self.layer_name} not found in model.")

    def remove_handlers(self):
        for h in self.handlers:
            h.remove()

    # ---------------- 主调用 ----------------
    def __call__(self, inputs, index=None):
        """
        Args:
            inputs: Tensor, [1,3,H,W]，已经在对应 device 上
            index:  目标类别 id；None 表示用当前预测类别

        Returns:
            cam: numpy array, [H,W]，0~1，已 resize 到 img_size×img_size
        """
        self.net.zero_grad()
        outputs = self.net(inputs)  # 可能是 logits 或 (logits, extra)

        # 兼容 (logits, extra) 这种返回
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs

        if index is None:
            index = int(torch.argmax(logits, dim=1).item())

        target_logit = logits[0, index]
        target_logit.backward()

        # 取 batch 维度 0 上的特征和梯度
        feat = self.feature[0].cpu()  # 形状可能是 [N,C] 或 [C,H,W]
        grad = self.gradient[0].cpu()

        # 兼容某些多一维的情况
        if feat.dim() == 3 and feat.shape[-1] == 1:
            feat = feat.squeeze(-1)
        if grad.dim() == 3 and grad.shape[-1] == 1:
            grad = grad.squeeze(-1)

        if feat.dim() == 2:
            # ViT token 特征: [N, C]
            cam = self._cam_from_vit_tokens(feat, grad)
        elif feat.dim() == 3:
            # CNN 特征: [C, H, W]
            cam = self._cam_from_cnn(feat, grad)
        else:
            raise ValueError(f"Unsupported feature shape: {feat.shape}")

        # resize 到 img_size×img_size
        cam = cv2.resize(cam, (self.img_size, self.img_size))
        return cam

    # ---------------- CNN 分支：普通 Grad-CAM ----------------
    def _cam_from_cnn(self, feat, grad):
        """
        feat: [C,H,W]
        grad: [C,H,W]
        """
        feat_np = feat.numpy()
        grad_np = grad.numpy()

        # 通道权重：对 H,W 做 GAP
        weight = np.mean(grad_np, axis=(1, 2))  # [C]
        cam = feat_np * weight[:, None, None]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam -= cam.min()
            cam /= cam.max()
        return cam

    # ---------------- ViT 分支：token -> patch map ----------------
    def _cam_from_vit_tokens(self, feat, grad):
        """
        feat: [N, C]   N = token 数 (可能 = 1 + num_patches)
        grad: [N, C]

        步骤：
          1) 对每个通道在 token 维做平均，得到通道权重 w_c
          2) 对每个 token 做 sum_c (feat[t,c] * w_c) 得到 token score
          3) 判断是否有 cls token（N-1 是完全平方数则认为有）
          4) 把 patch token reshape 成 H×W
        """
        feat_np = feat.numpy()
        grad_np = grad.numpy()
        N, C = feat_np.shape

        # 通道权重：对 token 维做 GAP
        weight = np.mean(grad_np, axis=0)  # [C]

        # 每个 token 的分数
        token_score = np.dot(feat_np, weight)  # [N]

        # 判断是否有 cls token
        if (N - 1) > 0 and int(np.sqrt(N - 1)) ** 2 == (N - 1):
            # 形如 ViT-B/16: N = 1 + 14*14 = 197
            patch_score = token_score[1:]  # 去掉 cls
            side = int(np.sqrt(N - 1))  # 14
        elif int(np.sqrt(N)) ** 2 == N:
            # 没有 cls token，直接 reshape
            patch_score = token_score
            side = int(np.sqrt(N))
        else:
            # 不规则情况：尽量 reshape 成近似方阵
            side = int(np.sqrt(N))
            patch_score = token_score[: side * side]

        cam = patch_score.reshape(side, side)
        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam -= cam.min()
            cam /= cam.max()
        return cam


# ------------------------ 二、工具函数：读图 & 叠加 CAM ------------------------


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_as_tensor(img_path, img_size=224, device="cuda"):
    from PIL import Image
    import torchvision.transforms as T
    import torch

    img = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # [0,1], 默认 float32
    ])
    tensor = transform(img).unsqueeze(0)  # [1,3,H,W]

    # 🔴 显式转成 float32，再丢到 GPU
    tensor = tensor.to(device=device, dtype=torch.float32)

    return tensor


def tensor_to_uint8_image(tensor, img_size=224):
    """
    把归一化后的 tensor 还原成 0~255 的 RGB 图像（用于可视化叠加）
    tensor: [1,3,H,W]
    """
    t = tensor[0].detach().cpu().numpy()  # [3,H,W]
    t = t.transpose(1, 2, 0)  # [H,W,3]
    # 反标准化
    t = t * np.array(IMAGENET_STD)[None, None, :] + np.array(IMAGENET_MEAN)[None, None, :]
    t = np.clip(t, 0.0, 1.0)
    img_uint8 = (t * 255).astype(np.uint8)
    return img_uint8


def overlay_cam_on_image(img_uint8, cam, alpha=0.4):
    """
    将 CAM 热力图叠加到原图上。
    img_uint8: 原图，numpy [H,W,3], uint8, 0~255
    cam:       CAM，numpy [H,W], 0~1
    alpha:     热力图透明度
    """
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转成 RGB

    # 融合
    overlay = (1 - alpha) * img_uint8.astype(np.float32) + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# ------------------------ 三、主流程示例 ------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ======= 1. 修改这里：你的 clean / adv 图像路径 =======
    clean_path = "/home/nan/dataset/ImageNet-compatible-dataset/images/1f3f949ccbc81c93.png"  # TODO: 改成你的 clean 图像路径
    adv_path = "/home/nan/dataset/output/mat/vit_base_patch16_224/1f3f949ccbc81c93.png"  # TODO: 改成你的 对抗样本 图像路径

    if not os.path.exists(clean_path):
        print(f"找不到 clean 图像: {clean_path}")
        print("请先把路径改成你本地的 clean 图像文件，比如: images/0001_clean.png")
        return
    if not os.path.exists(adv_path):
        print(f"找不到 adv 图像: {adv_path}")
        print("请先把路径改成你本地的 adv 图像文件，比如: images/0001_adv_mat.png")
        return

    # ======= 2. 加载模型（这里用 timm 的 ViT-B/16 举例） =======
    if timm is None:
        raise RuntimeError("请先安装 timm: pip install timm")

    # 源模型
    # model_name = "vit_base_patch16_224"
    # model_name = "pit_b_224"                # PiT-B
    #model_name = "visformer_small"           # visformer
    model_name = "swin_tiny_patch4_window7_224"  # swin

    model = timm.create_model(model_name, pretrained=True)
    model.to(device)
    model.eval()

    # 看一下网络层名字，确定我们要 hook 的层
    # 建议先打印一次，手动确认：
    for name, _ in model.named_modules():
        print(name)

    # layer_name = "blocks.11.norm1"  # vit
    # layer_name = "transformers.2.blocks.3.norm1"  # pit
    # layer_name = "stage3.3.norm1"  # visformer
    layer_name = "layers.3.blocks.1.mlp"  # swin

    cam_extractor = ViTGradCAM(model, layer_name=layer_name, img_size=224)

    # ======= 3. 读图并转 tensor =======
    x_clean = load_image_as_tensor(clean_path, img_size=224, device=device)
    x_adv = load_image_as_tensor(adv_path, img_size=224, device=device)

    # ======= 4. 计算 CAM（可以用预测类别，也可以指定 index=某个类别） =======
    cam_clean = cam_extractor(x_clean, index=None)  # numpy [224,224]
    cam_adv = cam_extractor(x_adv, index=None)

    # ======= 5. 还原原图，叠加 CAM 并保存 =======
    img_clean_uint8 = tensor_to_uint8_image(x_clean, img_size=224)
    img_adv_uint8 = tensor_to_uint8_image(x_adv, img_size=224)

    overlay_clean = overlay_cam_on_image(img_clean_uint8, cam_clean, alpha=0.4)
    overlay_adv = overlay_cam_on_image(img_adv_uint8, cam_adv, alpha=0.4)

    os.makedirs("cam_vis", exist_ok=True)
    clean_out = os.path.join("cam_vis", "clean_cam_swin.png")
    adv_out = os.path.join("cam_vis", "adv_cam_swin.png")

    # 用 cv2 保存（注意 cv2 用 BGR，所以要转一下）
    cv2.imwrite(clean_out, cv2.cvtColor(overlay_clean, cv2.COLOR_RGB2BGR))
    cv2.imwrite(adv_out, cv2.cvtColor(overlay_adv, cv2.COLOR_RGB2BGR))

    print("保存完成：")
    print("  ", clean_out)
    print("  ", adv_out)


if __name__ == "__main__":
    main()
