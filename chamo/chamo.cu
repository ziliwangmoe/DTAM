#include <opencv2/core/cuda/common.hpp>
#define size_block 32
#define t_block_size 5

__global__ void eff_test(float *in_data, float *gx, float *gy, int cols, int rows){
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x == 0 || y == 0 || x >= cols || y >= rows){
		return;
	}

	unsigned pt = y*rows + x;
	unsigned pt_l = pt - 1;
	unsigned pt_r = pt + 1;
	unsigned pt_u = pt - cols;
	unsigned pt_d = pt + cols;
	int i, j;
	float aa=0;
	int upper = cols*rows;
	int nn = 0;
	int start_y = 0;
	for (i = -t_block_size; i < t_block_size; i++){
		start_y = (y + i)*rows +x;
		for (j = -t_block_size; j < t_block_size; j++){
			nn = start_y + j;
			if (nn >= 0 && nn < upper){
				aa = aa + in_data[nn];
			}
		}
	}
	//int blockTol = t_block_size*rows;
	//int start_y = y*rows;
	//for (i = start_y - blockTol; i < start_y + blockTol; i = i + rows){
	//	int start_x = i+x;
	//	for (j = start_x - t_block_size; j < start_x + t_block_size; j++){
	//		if (j >= 0 && j < upper){
	//			aa = aa + in_data[j];
	//		}
	//	}
	//}

	gx[pt] = aa;
	//float aa=in_data[pt];

	//gx[pt] = in_data[pt_r] - in_data[pt_l];
	//gx[pt] = in_data[pt_r] - in_data[pt_l] - in_data[pt_l+3] + in_data[pt_l+6];
	//gx[pt] = __shfl_down(aa, 1) - __shfl_up(aa, 1) - __shfl_down(aa, 2) + __shfl_up(aa, 3);
	//gy[pt] = in_data[pt_d] - in_data[pt_u];
}

void eff_test_caller(float *in_data, float *gx, float *gy, int cols, int rows, cudaStream_t stream){
	dim3 dimBlock(size_block, size_block);
	dim3 dimGird((cols + dimBlock.x - 1) / (dimBlock.x), (rows + dimBlock.y - 1) / (dimBlock.y));
	for (int i = 0; i < 1; i++){
		eff_test << <dimGird, dimBlock, 0, stream >> >(in_data, gx, gy, cols, rows);
	}
}

__global__ void eff_tex_test(cudaTextureObject_t in_data, float *gx, float *gy, int cols, int rows){
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x == 0 || y == 0 || x >= cols || y >= rows){
		return;
	}
	unsigned pt = y*rows + x;
	//float p_l= tex2D<float>(in_data, x-1, y);
	//float p_r = tex2D<float>(in_data, x + 1, y);
	//float p_u = tex2D<float>(in_data, x, y);
	//float xx, yy;
	int i, j;
	float aa=0;
	//float ii = 400;
	for (i = y - t_block_size; i < y+t_block_size; i++){
		//yy = y + i;
		for (j = x - t_block_size; j < x+t_block_size; j++){
			//xx = x + j;
			//if (xx >= 0 && xx < cols && yy >= 0 && yy < rows){
			aa = aa + tex2D<float>(in_data, i, j);
			//}
			
		}
	}

	gx[pt] = aa;
}

void eff_test_tex_caller(cudaTextureObject_t in_data, float *gx, float *gy, int cols, int rows, cudaStream_t stream){
	dim3 dimBlock(size_block, size_block);
	dim3 dimGird((cols + dimBlock.x - 1) / (dimBlock.x), (rows + dimBlock.y - 1) / (dimBlock.y));
	for (int i = 0; i < 1; i++){
		eff_tex_test << <dimGird, dimBlock, 0, stream >> >(in_data, gx, gy, cols, rows);
	}
}