import torch
import cv2
import numpy as np
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, reparameter
import os

print(">>> 🚀 正在启动黑盒提点工具 (原图满血 FP16 极限版)...")

# ================= 1. 开启 FP16 极限模式 =================
_default_cfg = deepcopy(full_default_cfg)
_default_cfg['half'] = True # 【核心魔法】开启半精度，显存占用直接减半！

matcher = LoFTR(config=_default_cfg)
weight_path = "weights/eloftr_outdoor.ckpt"

if not os.path.exists(weight_path):
    print(f"❌ 找不到权重文件！请确保你已经把文件上传到了 {weight_path}")
    exit()

matcher.load_state_dict(torch.load(weight_path)['state_dict'])
matcher = reparameter(matcher)

# 将模型转换为 half() 并放入 GPU
matcher = matcher.half().eval().cuda()
print(">>> ✅ FP16 半精度模型加载成功，显存已释放一半！")

# ================= 2. 满血提取函数 =================
def extract_keypoints(img_path1, img_path2):
    # 读取原始高清灰度图
    img0_raw = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img0_raw is None or img1_raw is None:
        print("❌ 图片读取失败，请检查图片名字是否写对！")
        return None, None
        
    orig_h0, orig_w0 = img0_raw.shape
    orig_h1, orig_w1 = img1_raw.shape
    
    # 尺寸微调：绝不缩放，只做边缘裁剪 (裁掉除不尽32的余数像素)
    # 因为深度学习卷积层强制要求边长是32的倍数，最多只裁掉边缘几个像素，绝不压缩画质
    new_w0 = orig_w0 // 32 * 32
    new_h0 = orig_h0 // 32 * 32
    new_w1 = orig_w1 // 32 * 32
    new_h1 = orig_h1 // 32 * 32
    
    print(f"    [微调] 图1原图保持 {orig_w0}x{orig_h0} 清晰度 (仅切除边缘至 {new_w0}x{new_h0})")
    print(f"    [微调] 图2原图保持 {orig_w1}x{orig_h1} 清晰度 (仅切除边缘至 {new_w1}x{new_h1})")
    
    img0_raw = img0_raw[:new_h0, :new_w0]
    img1_raw = img1_raw[:new_h1, :new_w1]
    
    # 【核心】图片数据也必须转换为 half() 类型，匹配 FP16 模型
    img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
    batch = {'image0': img0, 'image1': img1}
    
    # 推理榨取 (原图对算力要求极高，这步可能需要几秒钟，耐心等待)
    with torch.no_grad():
        matcher(batch)
        
    # 提取坐标点
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
        
    return mkpts0, mkpts1

# ================= 3. 运行入口 =================
if __name__ == "__main__":
    image1 = "building_A.jpg"
    image2 = "building_B.jpg"
    
    if not os.path.exists(image1) or not os.path.exists(image2):
         print("❌ 找不到测试图片！请先上传图片到项目根目录。")
    else:
        print(f">>> 正在以【最高清晰度】处理图片: {image1} 和 {image2} ...")
        pts1, pts2 = extract_keypoints(image1, image2)
        
        if pts1 is not None:
            print(f">>> 🎉 满血榨取成功！一共提取到 {len(pts1)} 对超高精度匹配点。")
            # --- 新增的合并单文件代码 ---
            # 将图1和图2的坐标在水平方向拼接成 (N, 4) 的矩阵
            matches = np.hstack((pts1, pts2))
            # 导出一个干净的 txt 单文件，以空格分隔，保留 6 位小数
            txt_filename = 'custom_matches.txt'
            np.savetxt(txt_filename, matches, fmt='%.6f', delimiter=' ')
            
            print(f">>> 💾 实验数据合并成功！{len(matches)} 对特征点已保存至 {txt_filename} (格式: x1 y1 x2 y2)")