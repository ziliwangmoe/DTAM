#include <cuda_runtime.h>
#include "chamo.cuh"
#define BLOCKSIZE 32

__global__ void updataCV(cvInput input){
	int u = blockIdx.x*blockDim.x + threadIdx.x;
	int v = blockIdx.y*blockDim.y + threadIdx.y;
	if (u < input.width && v < input.height){
		int pt = v*input.width + u;
		float u_ref_t = input.transMat[0] * u + input.transMat[1] * v + input.transMat[2];
		float v_ref_t = input.transMat[4] * u + input.transMat[5] * v + input.transMat[6];
		float d_ref_t = input.transMat[8] * u + input.transMat[9] * v + input.transMat[10];
		float lowestErr = -1;
		float lowestErrInd = -1;
		float lowestD = -1;
		int steps = 0;
		for (float d = input.farZ; d < input.nearZ; d = d + input.zStep){
			float u_ref = u_ref_t + input.transMat[3] * d;
			float v_ref = v_ref_t + input.transMat[7] * d;
			float d_ref = d_ref_t + input.transMat[11] * d;
			u_ref = u_ref / d_ref;
			v_ref = v_ref / d_ref;
			float4 ref_val = tex2D<float4>(input.refImg, u_ref, v_ref);
			float v1 = fabsf(ref_val.x - input.baseImg[pt * 4]);
			float v2 = fabsf(ref_val.y - input.baseImg[pt * 4 + 1]);
			float v3 = fabsf(ref_val.z - input.baseImg[pt * 4 + 2]);
			//float err = fabsf(ref_val - input.baseImg[pt]);
			float oldErr = input.cvData[pt*input.stepCount + steps];
			float err = oldErr*(1 - input.weightPerImg) + (v1 + v2 + v3)*input.weightPerImg;
			//err = oldErr*(1-input.weightPerImg) + err*input.weightPerImg;
			input.cvData[pt*input.stepCount + steps] = err;
			if (lowestErr == -1){
				lowestErr = err;
				lowestErrInd = steps;
				lowestD = d;
			}
			else{
				if (lowestErr > err){
					lowestErr = err;
					lowestErrInd = steps;
					lowestD = d;
				}
			}
			steps++;
		}
		input.lowValue[pt] = lowestD;
		input.lowInd[pt] = lowestErrInd;
	}
}


void updataCVCaller(cvInput input){
	dim3 blockDim(BLOCKSIZE,BLOCKSIZE);
	dim3 girdDim((input.width + blockDim.x - 1) / (blockDim.x), (input.height + blockDim.y - 1) / (blockDim.y));
	updataCV << <girdDim, blockDim >> >(input);
}