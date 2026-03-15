import torch
import cv2
import numpy as np
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, reparameter
import os

print(">>> 🚀 正在启动黑盒提点工具 (写论文专属：4K防爆显存+坐标完美还原+TXT合并导出)...")

# ================= 1. 开启 FP16 极限模式 =================
_default_cfg = deepcopy(full_default_cfg)
_default_cfg['half'] = True 

matcher = LoFTR(config=_default_cfg)
weight_path = "weights/eloftr_outdoor.ckpt"

if not os.path.exists(weight_path):
    print(f"❌ 找不到权重文件！请确保文件已上传至 {weight_path}")
    exit()

matcher.load_state_dict(torch.load(weight_path)['state_dict'])
matcher = reparameter(matcher).half().eval().cuda()
print(">>> ✅ FP16 模型加载就绪！")

# ================= 2. 智能缩放 + 坐标反向放大 =================
def extract_keypoints(img_path1, img_path2, max_edge=2000):
    img0_raw = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img0_raw is None or img1_raw is None:
        print(f"❌ 图片读取失败，请检查文件名！")
        return None, None
        
    orig_h0, orig_w0 = img0_raw.shape
    orig_h1, orig_w1 = img1_raw.shape
    
    # 【核心防爆锁】：最大边长限制为 2000，且绝不放大原图 (末尾的 1.0 补丁)
    scale0 = min(max_edge / orig_w0, max_edge / orig_h0, 1.0)
    scale1 = min(max_edge / orig_w1, max_edge / orig_h1, 1.0)
    
    new_w0 = int(orig_w0 * scale0) // 32 * 32
    new_h0 = int(orig_h0 * scale0) // 32 * 32
    new_w1 = int(orig_w1 * scale1) // 32 * 32
    new_h1 = int(orig_h1 * scale1) // 32 * 32
    
    print(f"    [安全缩放] 图1从 {orig_w0}x{orig_h0} -> 压缩至 {new_w0}x{new_h0} 喂给模型")
    print(f"    [安全缩放] 图2从 {orig_w1}x{orig_h1} -> 压缩至 {new_w1}x{new_h1} 喂给模型")
    
    img0_resized = cv2.resize(img0_raw, (new_w0, new_h0))
    img1_resized = cv2.resize(img1_raw, (new_w1, new_h1))
    
    img0 = torch.from_numpy(img0_resized)[None][None].half().cuda() / 255.
    img1 = torch.from_numpy(img1_resized)[None][None].half().cuda() / 255.
    batch = {'image0': img0, 'image1': img1}
    
    with torch.no_grad():
        matcher(batch)
        
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    
    # 【核心还原魔法】：将提取到的坐标，精准放大回 4K 原图上的真实物理坐标
    if len(mkpts0) > 0:
        mkpts0[:, 0] = mkpts0[:, 0] * (orig_w0 / new_w0)
        mkpts0[:, 1] = mkpts0[:, 1] * (orig_h0 / new_h0)
        mkpts1[:, 0] = mkpts1[:, 0] * (orig_w1 / new_w1)
        mkpts1[:, 1] = mkpts1[:, 1] * (orig_h1 / new_h1)
        print("    [坐标魔法] 特征点坐标已成功 100% 映射回原图物理尺寸！")
        
    return mkpts0, mkpts1

# ================= 3. 运行与一键保存实验数据 =================
if __name__ == "__main__":
    image1 = "10.jpg"
    image2 = "45.jpg"
    
    if not os.path.exists(image1) or not os.path.exists(image2):
         print("❌ 找不到测试图片！")
    else:
        print(f">>> 正在处理: {image1} & {image2} ...")
        pts1, pts2 = extract_keypoints(image1, image2)
        
        if pts1 is not None and len(pts1) > 0:
            # 横向合并为 (N, 4) 的矩阵
            matches = np.hstack((pts1, pts2))
            
            # 自动生成实验数据文件名
            txt_filename = 'custom_matches.txt'
            np.savetxt(txt_filename, matches, fmt='%.6f', delimiter=' ')
            print(f">>> 🎉 实验数据合并成功！{len(matches)} 对特征点已保存至 {txt_filename} (格式: x1 y1 x2 y2)")