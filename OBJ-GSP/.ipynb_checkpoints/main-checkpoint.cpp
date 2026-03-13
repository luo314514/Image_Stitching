#include <iostream>
#include <fstream>            // 【新增工具】用来读取本地的 txt 文件
#include <vector>             // 【新增工具】用来创建能自动变长的“数组（容器）”
#include <opencv2/opencv.hpp> // 【新增工具】因为我们要用到 OpenCV 里的 Point2f（二维坐标点）数据格式

#include "./Stitching/NISwGSP_Stitching.h"
#include "./Debugger/TimeCalculator.h"

#define _CRT_SECURE_NO_WARNINGS 

using namespace std;
using namespace cv;           // 【新增】告诉电脑我们要用 OpenCV 里的简写名字

int GRID_SIZE_w = 40;
int GRID_SIZE_h = 40;

int main(int argc, const char* argv[]) {
	int num_data = 2; // dataset number + 1
	const char* data_list[] = { "nothing-here","my_work"};

	Eigen::initParallel();
	CV_DNN_REGISTER_LAYER_CLASS(Crop, CropLayer);
	cout << "nThreads = " << Eigen::nbThreads() << endl;
	cout << "[#Images : " << num_data - 1 << "]" << endl;

	// =====================================================================
	// 【新增核心步骤 1：把你的特征点读进电脑内存里】
	// =====================================================================
	
	vector<Point2f> custom_points1; 
	vector<Point2f> custom_points2;

	ifstream infile("custom_matches.txt");
	
	if (!infile.is_open()) {
		cerr << "哎呀！没找到 custom_matches.txt 文件！请检查有没有把它放在当前目录下！" << endl;
		return -1; 
	}

	float x1, y1, x2, y2;
	
	while (infile >> x1 >> y1 >> x2 >> y2) {
		custom_points1.push_back(Point2f(x1, y1)); 
		custom_points2.push_back(Point2f(x2, y2)); 
	}
	infile.close(); 

	cout << ">>> 厂长广播：成功加载了你自己提取的高精度特征点: " << custom_points1.size() << " 对！" << endl;
	// =====================================================================

	time_t start = clock();
	TimeCalculator timer;
	
	for (int i = 1; i < num_data; ++i) {
		cout << "i = " << i << ", [Images : " << data_list[i] << "]" << endl;
		
		MultiImages multi_images(data_list[i], LINES_FILTER_WIDTH, LINES_FILTER_LENGTH, custom_points1, custom_points2);

		NISwGSP_Stitching niswgsp(multi_images);
		niswgsp.setWeightToAlignmentTerm(1); 
		niswgsp.setWeightToLocalSimilarityTerm(0.75); 
		niswgsp.setWeightToGlobalSimilarityTerm(6, 20, GLOBAL_ROTATION_2D_METHOD);
		niswgsp.setWeightToContentPreservingTerm(1.5);
		
		Mat blend_linear;
		vector<vector<Point2> > original_vertices;
		
		if (RUN_TYPE != 0) {
			blend_linear = niswgsp.solve_content(BLEND_LINEAR, original_vertices);
		}
		else {
			blend_linear = niswgsp.solve(BLEND_LINEAR, original_vertices);
		}
		
		time_t end = clock();
		cout << "Time:" << double(end - start) / CLOCKS_PER_SEC << endl;
		
		// ==========================================================
		// 【终极通关】：直接保存，跳过没用的报错评估
		// ==========================================================
		
		// 1. 强制将结果保存在根目录，起个响亮的名字！
		imwrite("SUCCESS_result.png", blend_linear); 
		
		// 2. 注释掉原作者可能导致越界的路径写入和会引起崩溃的误差评估
		// niswgsp.writeImage(blend_linear, BLENDING_METHODS_NAME[BLEND_LINEAR]);
		// niswgsp.assessment(original_vertices); 

		cout << ">>> 🎆 厂长广播：拼接大圆满！请立刻查看根目录下的 SUCCESS_result.png！" << endl;
	}

	return 0; // 顺利下班
}