# 🚀 Hybrid-Image-Stitching (端到端混合驱动图像拼接引擎)

This repository proposes an End-to-End (E2E) Hybrid Image Stitching Pipeline that combines the robustness of Deep Learning (DL) with the geometric accuracy of traditional optimization algorithms.
本项目实现了一个端到端的混合图像拼接引擎，完美融合了深度学习特征提取的鲁棒性与传统几何优化算法的高精度。

## 🌟 Core Architecture (核心架构)

The pipeline is decoupled into two powerful modules:
本项目分为两个核心模块协同工作：

1. **Frontend: AI Perception (`efficientloftr`)**
   - Extracts highly accurate, dense feature matching points using DL-based methods (LoFTR).
   - Solves traditional algorithms' failure cases in low-texture regions, repetitive patterns (e.g., glass buildings), and complex illuminations.
   - **前端（AI 感知）**：摒弃传统 SIFT/SURF，使用深度学习密集匹配技术，提取成千上万的高精度点云，完美解决玻璃大楼、水面反光等无特征或重复纹理区域的匹配死穴。

2. **Backend: Geometric Optimization (`OBJ-GSP`)**
   - Consumes the dense feature points and constructs a global mesh.
   - Utilizes **APAP (As-Projective-As-Possible)** for local mesh deformation to perfectly align severe parallax regions.
   - Utilizes **GSP (Global Similarity Prior)** to constrain non-overlapping areas, preventing perspective distortions and ensuring natural shape preservation.
   - **后端（网格约束与优化）**：接管点云，构建巨型网格。利用 **APAP** 算法在重叠区进行局部强力拉伸扭曲（完美消除大视差重影），同时利用 **GSP** 全局相似性约束保护非重叠区（防止大楼边缘被过度拉扯变形）。

## 💡 Key Features (核心亮点)

- **4K High-Resolution Support (超高清支持)**: Unlocked hard-coded downsampling limits, allowing raw 4K/12MP image ingestion.
- **Parametric RANSAC Control (参数化安检阀)**: 
  - Strict mode (`RANSAC=1.5, Step=10`): Filters AI hallucinations (e.g., water reflections) for planar structures.
  - Parallax mode (`RANSAC=15.0, Step=1`): Saturation injection for handling massive foreground-background parallax.
- **Math-Safe (数学防爆)**: Hardcoded focal scale locking to prevent `NaN` explosions in complex transformations.

## 🛠️ Usage (如何使用)
1. Run `efficientloftr` to extract feature pairs and export `custom_matches.txt`.
2. Place the images in `OBJ-GSP/input-data/` and modify the graph txt file.
3. Compile the C++ backend: `cd OBJ-GSP/build && cmake .. && make -j6`
4. Run `./obj_gsp` to get the flawless stitching result!
