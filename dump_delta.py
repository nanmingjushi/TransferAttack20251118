"""
  @Project ：bfa.py 
  @File ：dump_delta.py
  @IDE ：PyCharm 
  @Author ：26354
  @Date ：2025/12/14 18:37 
"""


# 生成扰动图像  adv减去clean


import os
import numpy as np
from PIL import Image

# =========================
# Utilities
# =========================

def load_rgb(path):
    return Image.open(path).convert("RGB")

def to_float01(img):
    return np.asarray(img).astype(np.float32) / 255.0

def save_uint8(path, arr01):
    arr01 = np.clip(arr01, 0.0, 1.0)
    arr8 = (arr01 * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(path)

# =========================
# Main
# =========================

def main(clean_path, adv_path, out_dir="fig_paper_cnn"):
    os.makedirs(out_dir, exist_ok=True)

    clean_img = load_rgb(clean_path)
    adv_img   = load_rgb(adv_path)

    # 论文中：尺寸必须一致（以模型输入为准）
    if clean_img.size != adv_img.size:
        clean_img = clean_img.resize(adv_img.size, Image.BILINEAR)

    clean = to_float01(clean_img)
    adv   = to_float01(adv_img)

    # 扰动
    delta = adv - clean

    # ========== (a) Clean ==========
    save_uint8(os.path.join(out_dir, "fig_a_clean.png"), clean)

    # ========== (b) Adversarial ==========
    save_uint8(os.path.join(out_dir, "fig_b_adv.png"), adv)

    # ========== (c) Perturbation (normalized) ==========
    # Normalize by max absolute value (paper standard)
    max_abs = np.max(np.abs(delta)) + 1e-12
    delta_norm = delta / max_abs
    delta_vis = delta_norm * 0.5 + 0.5
    save_uint8(os.path.join(out_dir, "fig_c_delta_norm.png"), delta_vis)

    # ========== (d) Perturbation magnitude ==========
    # L2 magnitude per pixel
    mag = np.sqrt(np.sum(delta ** 2, axis=2))
    mag = mag / (np.max(mag) + 1e-12)
    mag_vis = np.stack([mag, mag, mag], axis=2)
    save_uint8(os.path.join(out_dir, "fig_d_delta_mag.png"), mag_vis)

    # ========== Metrics (for caption / appendix) ==========
    linf = np.max(np.abs(delta))
    l2   = np.sqrt(np.sum(delta ** 2))
    mean = np.mean(np.abs(delta))

    print("=== Perturbation statistics (pixel space) ===")
    print(f"L_inf : {linf:.6f}  (~{linf*255:.2f}/255)")
    print(f"L2    : {l2:.6f}")
    print(f"Mean  : {mean:.6f}  (~{mean*255:.2f}/255)")
    print(f"Saved figures to: {out_dir}")
    print("  fig_a_clean.png")
    print("  fig_b_adv.png")
    print("  fig_c_delta_norm.png  (normalized for visualization)")
    print("  fig_d_delta_mag.png")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="path to clean PNG")
    ap.add_argument("--adv", required=True, help="path to adversarial PNG")
    ap.add_argument("--out", default="fig_paper", help="output directory")
    args = ap.parse_args()
    main(args.clean, args.adv, args.out)


# 生成扰动图像  adv减去clean
# python dump_delta.py --clean D:\000-dataset\ImageNet-compatible-dataset\images\0e0f1fd2ed183781.png --adv D:\000-dataset\output\mat\vit_base_patch16_224\0e0f1fd2ed183781.png --out out_delta_vit