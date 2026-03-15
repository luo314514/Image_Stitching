import os
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, reparameter
from src.utils.plotting import make_matching_figure

print(">>> 🎨 正在启动超清连线图一键绘制工具 (带精简显示开关)...")

# ================= 1. 加载 FP16 满血模型 =================
_default_cfg = deepcopy(full_default_cfg)
_default_cfg['half'] = True

matcher = LoFTR(config=_default_cfg)
weight_path = "weights/eloftr_outdoor.ckpt" 

if not os.path.exists(weight_path):
    print(f"❌ 找不到权重文件 {weight_path}，请检查路径！")
    exit()

matcher.load_state_dict(torch.load(weight_path)['state_dict'])
matcher = reparameter(matcher).half().eval().cuda()
print(">>> ✅ 模型就绪，准备作图！")

# ================= 2. 核心提取与抽样绘图逻辑 =================
def draw_matches(img_path1, img_path2, max_edge=2000, draw_ratio=0.01):
    """
    draw_ratio: 绘图时的连线抽样比例。
                0.01 表示只画 1% 的线（适合 5000+ 对点的情况，让人眼能看清图）
                1.0 表示 100% 全画
    """
    img0_raw = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img0_raw is None or img1_raw is None:
        print(f"❌ 图片读取失败，请检查: {img_path1} 和 {img_path2}")
        return

    # 防爆显存安全锁
    scale0 = min(max_edge / img0_raw.shape[1], max_edge / img0_raw.shape[0], 1.0)
    scale1 = min(max_edge / img1_raw.shape[1], max_edge / img1_raw.shape[0], 1.0)
    
    new_w0 = int(img0_raw.shape[1] * scale0) // 32 * 32
    new_h0 = int(img0_raw.shape[0] * scale0) // 32 * 32
    new_w1 = int(img1_raw.shape[1] * scale1) // 32 * 32
    new_h1 = int(img1_raw.shape[0] * scale1) // 32 * 32
    
    img0_resized = cv2.resize(img0_raw, (new_w0, new_h0))
    img1_resized = cv2.resize(img1_raw, (new_w1, new_h1))
    
    img0 = torch.from_numpy(img0_resized)[None][None].half().cuda() / 255.
    img1 = torch.from_numpy(img1_resized)[None][None].half().cuda() / 255.
    batch = {'image0': img0, 'image1': img1}
    
    print(f"    [计算中] 正在提取 {img_path1} & {img_path2} 的特征点...")
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    total_matches = len(mkpts0)
    if total_matches == 0:
        print("⚠️ 未找到任何匹配点！")
        return

    # ================= 3. 【新增】连线降采样开关 =================
    if draw_ratio < 1.0:
        num_draw = max(1, int(total_matches * draw_ratio))
        print(f"    [视觉优化] 提取到 {total_matches} 对点，为避免画面拥挤，随机抽取 {num_draw} 对 ({draw_ratio*100}%) 进行连线绘制...")
        
        # 随机抽取指定数量的索引
        indices = np.random.choice(total_matches, num_draw, replace=False)
        mkpts0_draw = mkpts0[indices]
        mkpts1_draw = mkpts1[indices]
        mconf_draw = mconf[indices]
    else:
        print(f"    [渲染中] 准备绘制全部 {total_matches} 对匹配点...")
        mkpts0_draw = mkpts0
        mkpts1_draw = mkpts1
        mconf_draw = mconf

    # ================= 4. 渲染并导出超清图片 =================
    color = cm.jet(mconf_draw)
    text = ['LoFTR', f'Matches: {total_matches} (Drawn: {len(mkpts0_draw)})']
    
    fig = make_matching_figure(img0_resized, img1_resized, mkpts0_draw, mkpts1_draw, color, text=text)
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    img1_name = os.path.splitext(os.path.basename(img_path1))[0]
    img2_name = os.path.splitext(os.path.basename(img_path2))[0]
    # 在文件名里标明抽样比例，防止和之前画的全连线图弄混
    save_path = os.path.join(save_dir, f"match_{img1_name}_{img2_name}_drawn{int(draw_ratio*100)}pct.jpg")
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">>> 🎆 大功告成！清爽版连线图已保存至: {save_path}\n")

# ================= 5. 执行入口 =================
if __name__ == "__main__":
    image_A = "10.jpg"
    image_B = "45.jpg"
    
    # 核心开关在这里！
    # draw_ratio=0.01 表示只画 1%，改为 1.0 表示全画
    draw_matches(image_A, image_B, draw_ratio=0.01)