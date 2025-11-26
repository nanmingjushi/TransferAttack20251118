"""
  @Project ：TransferAttack20251118
  @File    ：download_models.py
  @IDE     ：PyCharm
  @Author  ：26354
  @Date    ：2025/11/26 15:32
"""

# 统一下载模型（使用 torchvision 新权重系统：weights=DEFAULT）

import timm
import torchvision.models as tvm
from torchvision.models._api import WeightsEnum

cnn_model_paper = ['resnet50', 'vgg16', 'mobilenet_v2', 'inception_v3']
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224',
                   'visformer_small', 'swin_tiny_patch4_window7_224']

cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']

vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

all_models = list(dict.fromkeys(cnn_model_paper + vit_model_paper + cnn_model_pkg + vit_model_pkg))

print("将下载以下模型：")
print(all_models)
print("\n开始下载...\n")

for name in all_models:
    print(f"Downloading {name} ...")
    try:
        if name in tvm.__dict__:   # torchvision 模型
            model_fn = tvm.__dict__[name]

            # 自动获取 WeightsEnum（新版 API 必须使用枚举）
            weights_enum = model_fn.__annotations__.get("weights", None)

            if isinstance(weights_enum, type) and issubclass(weights_enum, WeightsEnum):
                # 使用最新版 DEFAULT 权重
                model_fn(weights=weights_enum.DEFAULT)
            else:
                # 某些旧模型（极少），可能没有 weights 注解，退回旧接口
                model_fn(pretrained=True)

        else:
            # timm 模型
            timm.create_model(name, pretrained=True)

        print(f"成功：{name}\n")

    except Exception as e:
        print(f"失败：{name} -- {e}\n")

print("\n全部模型处理完毕！")
