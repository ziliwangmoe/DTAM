#ifndef CHAMO_CUH
#define CHAMO_CUH

#include <opencv2/core/cuda/common.hpp>
#include "../utils/utils.hpp"

struct cvInput{
	float *baseImg;
	float *cvData;
	cudaTextureObject_t refImg;
	float *lowInd;
	float *lowValue;
	float transMat[12];
	int width;
	int height;
	float nearZ;
	float farZ;
	float zStep;
	int stepCount;
	float weightPerImg;
};

struct cvIGradient{
	float *baseImg;
	float *graImg;
	float alpha;
	int width;
	int height;
};

struct updataQInput{
	float *invDepth;
	float *g;
	float *qx;
	float *qy;
	float *a;
	float sigma_d;
	float sigma_q;
	float eps;
	float theta;
	float *temp1;
	float *temp2;
	float *temp3;
	int width;
	int height;
};

struct updataAInput{
	float *invDepth;
	float *cvData;
	float *a;
	float stepSize;
	float startD;
	float lamda;
	int layers;
	int width;
	int height;
};


void eff_test_caller(float *in_data, float *gx, float *gy, int cols, int rows, cudaStream_t stream);
void eff_test_tex_caller(cudaTextureObject_t in_data, float *gx, float *gy, int cols, int rows, cudaStream_t stream);
void updataCVCaller(cvInput input);
void gradientCaller(cvIGradient input);
void updataQCaller(updataQInput input);
void updataACaller(updataAInput input);
#endif