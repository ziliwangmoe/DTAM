#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h> 

#include "chamo.cuh"
#include "../utils/utils.hpp"

using namespace cv;
using namespace std;
#define res_size 512
void efficience_try(){
	cuda::Stream cvStream;
	cudaEvent_t e1, e2, e3;
	cudaEventCreate(&e1);
	cudaEventCreate(&e2);
	cudaEventCreate(&e3);
	Mat res = Mat::eye(Size(res_size, res_size), CV_32FC1);
	clock_t t1 = clock();
	cuda::GpuMat g_res(Size(res_size, res_size), CV_32FC1);
	clock_t t2 = clock();
	g_res.upload(res, cvStream);
	clock_t t3 = clock();
	cuda::GpuMat gx(Size(res_size, res_size), CV_32FC1);
	cuda::GpuMat gy(Size(res_size, res_size), CV_32FC1);
	clock_t t4 = clock();
	cudaEventRecord(e1, cuda::StreamAccessor::getStream(cvStream));
	eff_test_caller((float *)g_res.data, (float *)gx.data, (float *)gy.data, res_size, res_size, cuda::StreamAccessor::getStream(cvStream));
	cudaEventRecord(e2, cuda::StreamAccessor::getStream(cvStream));
	clock_t t5 = clock();
	cudaDeviceSynchronize();
	clock_t t6 = clock();
	cudaSafeCall(cudaGetLastError());
	//cout << ((float)t5 - (float)t1) / CLOCKS_PER_SEC << endl;
	//cout << ((float)t6 - (float)t1)/CLOCKS_PER_SEC << endl;
	float time;
	cudaEventElapsedTime(&time, e1, e2);
	cout << time << endl;
	//showMat(gx);

	int aa = 1;
}

void efficience_try_tex(){
	cuda::Stream cvStream;
	cudaEvent_t e1, e2;
	cudaEventCreate(&e1);
	cudaEventCreate(&e2);
	Mat res = Mat::eye(Size(res_size, res_size), CV_32FC1);

	cudaArray* cuArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, res_size*sizeof(float), res_size));
	cout << res.dataend - res.datastart << endl;
	cudaMemcpyToArray(cuArray, 0, 0, res.datastart, res.dataend - res.datastart, cudaMemcpyHostToDevice);
	cudaSafeCall(cudaGetLastError());
	// Specify texture memory location
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureObject_t texObj;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaSafeCall(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

	cuda::GpuMat gx(Size(res_size, res_size), CV_32FC1);
	cuda::GpuMat gy(Size(res_size, res_size), CV_32FC1);
	cudaEventRecord(e1, cuda::StreamAccessor::getStream(cvStream));
	eff_test_tex_caller(texObj, (float *)gx.data, (float *)gy.data, res_size, res_size, cuda::StreamAccessor::getStream(cvStream));
	cudaEventRecord(e2, cuda::StreamAccessor::getStream(cvStream));
	cudaDeviceSynchronize();
	cudaSafeCall(cudaGetLastError());
	float time;
	cudaEventElapsedTime(&time, e1, e2);
	cout << time << endl;
	//showMat(gx);
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(cuArray);
}