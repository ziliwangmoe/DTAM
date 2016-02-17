#include "costVol_chamo.h"
#include <opencv2/cudaimgproc.hpp>


costVol_chamo::costVol_chamo(Mat _baseImg, int layerCount, float near, float far, Mat projMat, Mat camMat){
	layers = layerCount;
	width = _baseImg.cols;
	height = _baseImg.rows;
	nearZ = near;
	farZ = far;
	lowInd.create(Size(width, height), CV_32FC1);
	topVal.create(Size(width, height), CV_32FC1);
	bottonVal.create(Size(width, height), CV_32FC1);
	cvData.create(Size(1, height*width*layers), CV_32FC1);
	cvData.setTo(0);

	assert(lowInd.isContinuous());
	camIntrin = camMat;
	baseImgGray.upload(_baseImg);
	cuda::cvtColor(baseImgGray, baseImgGray, COLOR_BGRA2GRAY);
	//cuda::GpuMat gmat;
	//baseImgGray.convertTo(gmat, CV_32FC1);
	//showLenghtOfMat(gmat);
	//showMat(gmat);
	assert(baseImgGray.isContinuous());
	_baseImg.convertTo(_baseImg, CV_32FC4, 1 / 255.0);
	baseImg.upload(_baseImg);
	baseImgProjMat = projMat;
	assert(baseImg.isContinuous());
}

void costVol_chamo::updataCV(Mat refImg, Mat projMat, float weightPerImg){
	
	
	cudaArray* cuArray;
	cudaTextureObject_t texObj;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));

	cudaMemcpyToArray(cuArray, 0, 0, refImg.data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaSafeCall(cudaGetLastError());

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 0;
	cudaSafeCall(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

	Mat finalTran = projMat*baseImgProjMat.inv();
	cvInput input;
	input.baseImg = (float *)baseImg.data;
	input.cvData = (float*)cvData.data;
	input.nearZ = nearZ;
	input.farZ = farZ;
	input.height = height;
	input.width = width;
	input.lowInd = (float*)lowInd.data;
	input.lowValue = (float*)bottonVal.data;
	for (int i = 0; i < 12; i++){
		input.transMat[i] = finalTran.at<float>(i);
	}
	input.refImg = texObj;
	input.zStep = (nearZ - farZ) / layers;
	input.stepCount = layers;
	updataCount++;
	input.weightPerImg = 1.0 / updataCount;
	updataCVCaller(input);
}

void costVol_chamo::getGradientImg(){
	qx.create(height, width, CV_32FC1);
	qy.create(height, width, CV_32FC1);
	a.create(height, width, CV_32FC1);
	g.create(height, width, CV_32FC1);
	d.create(height, width, CV_32FC1);
	temp1.create(height, width, CV_32FC1);
	temp2.create(height, width, CV_32FC1);
	temp3.create(height, width, CV_32FC1);

	qx.setTo(0);
	qy.setTo(0);
	bottonVal.copyTo(a);
	bottonVal.copyTo(d);

	cvIGradient input;
	input.baseImg = (float *)baseImgGray.data;
	input.graImg = (float *)g.data;
	input.alpha = alpha;
	input.height = height;
	input.width = width;
	gradientCaller(input);
	
}

void costVol_chamo::updataD(){
	calIterStepSize();
	updataQInput input;
	input.g = (float *)g.data;
	input.a = (float *)a.data;
	input.invDepth = (float *)d.data;
	input.qx = (float *)qx.data;
	input.qy = (float *)qy.data;
	input.temp1 = (float *)temp1.data;
	input.temp2 = (float *)temp2.data;
	input.temp3 = (float *)temp3.data;
	input.height = height;
	input.width = width;
	input.sigma_d = sigma_d;
	input.sigma_q = sigma_q;
	input.eps = epsilon;
	input.theta = theta;
	updataQCaller(input);

}

void costVol_chamo::updataA(){
	updataAInput input;
	input.a = (float*)a.data;
	input.cvData = (float*)cvData.data;
	input.height = height;
	input.width = width;
	input.invDepth = (float *)d.data;
	input.lamda = lamda*theta * 2;
	input.layers = layers;
	input.startD = farZ;
	input.stepSize = (nearZ - farZ) / layers;
	updataACaller(input);
	if (true/*theta > thetaMin*/){
		theta = theta*thetaStep;
	}
}

void costVol_chamo::calIterStepSize(){
	float lambda_t, alpha, gamma, delta, mu, rho, sigma;
	float L = 4;//lower is better(longer steps), but in theory only >=4 is guaranteed to converge. For the adventurous, set to 2 or 1.44

	lambda_t = 1.0 / theta;
	alpha = epsilon;

	gamma = lambda_t;
	delta = alpha;

	mu = 2.0*sqrt(gamma*delta) / L;

	rho = mu / (2.0*gamma);
	sigma = mu / (2.0*delta);

	sigma_d = rho;
	sigma_q = sigma;
}