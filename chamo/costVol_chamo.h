#ifndef COSTVOL_CHAMO
#define COSTVOL_CHAMO

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <iostream>
#include <stdio.h>
#include <string>

#include "../utils/utils.hpp"
#include "chamo.cuh"

using namespace std;
using namespace cv;

class costVol_chamo{
public:
	costVol_chamo(Mat _baseImg, int layerCount, float near, float far, Mat projMat ,Mat camMat);
	void updataCV(Mat refImg, Mat projMat, float weightPerImg);
	void getGradientImg();
	void updataD();
	void updataA();

	cuda::GpuMat g;
	cuda::GpuMat a;
	cuda::GpuMat qx;
	cuda::GpuMat qy;
	cuda::GpuMat d;
	
	cuda::GpuMat baseImg; 	
	cuda::GpuMat baseImgGray;
	cuda::GpuMat lowInd;
	cuda::GpuMat topVal;
	cuda::GpuMat bottonVal;
private:
	int updataCount = 0;
	int layers = 32;
	int width, height;
	float nearZ, farZ;
	float lamda = 0.000001;
	float theta = 20;
	float thetaMin = 1.0;
	float thetaStep = .97;
	float epsilon = .1;
	float sigma_d;
	float sigma_q;
	float alpha = 3.5;

	Mat baseImgProjMat;
	Mat camIntrin;

	cuda::GpuMat cvData;
	cuda::GpuMat temp1;
	cuda::GpuMat temp2;
	cuda::GpuMat temp3;

	

	void calIterStepSize();
};

#endif