#include <cuda_runtime.h>
#include "chamo.cuh"
#define BLOCKSIZE 32

__global__ void getGradient(cvIGradient input){
	int u = blockIdx.x*blockDim.x + threadIdx.x;
	int v = blockIdx.y*blockDim.y + threadIdx.y;
	int pt = v*input.width + u;
	if (u > 0 && v > 0 && u < input.width - 1 && v < input.height - 1){
		float gx_t = fabsf(input.baseImg[pt - 1] - input.baseImg[pt + 1]);
		float gy_t = fabsf(input.baseImg[pt - input.width] - input.baseImg[pt + input.width]);
		float g_t = max(gx_t, gy_t);
		g_t = sqrt(g_t);
		input.graImg[pt] = gx_t/*exp(-input.alpha*g_t)*/;
	}
}

void gradientCaller(cvIGradient input){
	dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
	dim3 girdDim((input.width + blockDim.x - 1) / (blockDim.x), (input.height + blockDim.y - 1) / (blockDim.y));
	getGradient << <girdDim, blockDim >> >(input);
}


__global__ void updataQ(updataQInput input){
	int u = blockIdx.x*blockDim.x + threadIdx.x;
	int v = blockIdx.y*blockDim.y + threadIdx.y;
	int interBlockId = threadIdx.y*BLOCKSIZE + threadIdx.x;
	int pt = v*input.width + u;
	extern __shared__ float s[];
	if (u > 0 && v > 0 && u < input.width - 1 && v < input.height - 1){
		//input.temp1[pt] = input.invDepth[pt] * input.g[pt];
		//__syncthreads();
		float qx = ((input.invDepth[pt + 1] - input.invDepth[pt - 1])*input.sigma_q + input.qx[pt]) / (1 + input.sigma_q*input.eps);
		float qy = ((input.invDepth[pt + input.width] - input.invDepth[pt - input.width])*input.sigma_q + input.qy[pt]) / (1 + input.sigma_q*input.eps);
		float q_norm = 1/fmaxf(1.0f, sqrt(qx*qx + qy*qy));
		qx = qx * q_norm;
		qy = qy * q_norm;

		input.qx[pt] = qx;
		input.qy[pt] = qy;
		//__syncthreads();
		//s[interBlockId] = input.qx[pt + 1] - input.qx[pt - 1] + input.qy[pt + input.width] - input.qy[pt - input.width];
		__syncthreads();
		float temp = input.qx[pt + 1] - input.qx[pt - 1] + input.qy[pt + input.width] - input.qy[pt - input.width];
		//s[interBlockId] = s[interBlockId] * input.g[pt];
		input.invDepth[pt] = (input.invDepth[pt] + input.eps*(temp + input.a[pt] / input.theta)) / (1 + input.sigma_d / input.theta);
	}
}

void updataQCaller(updataQInput input){
	dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
	dim3 girdDim((input.width + blockDim.x - 1) / (blockDim.x), (input.height + blockDim.y - 1) / (blockDim.y));
	updataQ << <girdDim, blockDim, BLOCKSIZE*BLOCKSIZE*sizeof(float)*3 >> >(input);
}

__global__ void updataA(updataAInput input){
	int u = blockIdx.x*blockDim.x + threadIdx.x;
	int v = blockIdx.y*blockDim.y + threadIdx.y;
	int pt = v*input.width + u;
	
	
	float minErr = 1000;
	float minD = -1;
	float endD = input.stepSize*input.layers + input.startD;
	int n = 0;
	for (float i = input.startD; i < endD; i = i + input.stepSize){
		float temp = input.invDepth[pt] - i;
		temp = temp*temp;
		if (u == 200 && v == 200){
			printf("%.10f,%.10f\n", input.lamda*input.cvData[pt*input.layers + n], temp);
		}
		float curErr = input.lamda*input.cvData[pt*input.layers + n] + temp;
		if (minErr > curErr){
			minErr = curErr;
			minD = i;
		}
		n++;
	}
	input.a[pt] = minD*input.stepSize + input.startD;
}

void updataACaller(updataAInput input){
	dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
	dim3 girdDim((input.width + blockDim.x - 1) / (blockDim.x), (input.height + blockDim.y - 1) / (blockDim.y));
	updataA << <girdDim, blockDim>> >(input);
}