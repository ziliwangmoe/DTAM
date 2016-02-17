#include "imgInputer.h"
#include "../convertAhandaPovRayToStandard.h"
void imgInputerChamo::import(int count){
	string _dir = "D:/cuda_try/build/denseImg/";
	intrinMat = (Mat_<float>(3, 3) << 320, 0.0, 320,
		0.0, 320, 240,
		0.0, 0.0, 1.0);
	intrinMat = make4x4(intrinMat);
	float amp = 0.1;
	Mat camX = (Mat_<float>(9, 1) << 0, 1, -1, 0, 1, -1, 0, 1, -1);
	camX = camX*amp;
	Mat camY = (Mat_<float>(9, 1) << 0, 0, 1, 1, 1, -1, -1, -1, 0);
	camY = camY*amp;
	float camZ = -3;
	dir = _dir;
	projMats.reserve(count);
	imgList.reserve(count);
	imgCount = count;
	for (int i = 0; i < count; i++){
		Mat img;
		string filename = dir;
		char tempDigitStr[20];
		sprintf(tempDigitStr, "den%02d.jpg", i);
		filename = filename + tempDigitStr;
		img = imread(filename);
		width = img.cols;
		height = img.rows;
		imgList.push_back(img.clone());
		Mat viewMat = Mat::eye(4, 4, CV_32FC1);
		viewMat.at<float>(0, 3) = -camX.at<float>(i);
		viewMat.at<float>(1, 3) = -camY.at<float>(i);
		viewMat.at<float>(2, 3) = camZ;
		viewMat = intrinMat*viewMat;
		projMats.push_back(viewMat.clone());
	}
}

void imgInputerAhanda::import(int count){
	
	string _dir = "D:/OpenDTAM-2.4.9_experimental/OpenDTAM-2.4.9_experimental/Trajectory_30_seconds/";
	intrinMat = (Mat_<float>(3, 3) << 481.20, 0.0, 319.5,
		0.0, 480.0, 239.5,
		0.0, 0.0, 1.0);
	intrinMat = make4x4(intrinMat);
	for (int i = 0; i<count; i++){
		Mat R, T;
		convertAhandaPovRayToStandard(_dir.c_str(), i, R, T);
	    Mat img;
		string filename = _dir;
		char tempDigitStr[20];
		sprintf(tempDigitStr, "scene_%03d.png", i);
		filename = filename + tempDigitStr;
		img =imread(filename);
		cvtColor(img, img, COLOR_BGR2BGRA);
		//cout << (img.dataend - img.datastart) / img.cols / img.rows << endl;
		//imshow("csd", img);
		//waitKey(-1);
		imgList.push_back(img.clone());
		Mat viewMat;
		RTToP(R, T, viewMat);
		viewMat = intrinMat*viewMat;
		projMats.push_back(viewMat.clone());
	  }
}