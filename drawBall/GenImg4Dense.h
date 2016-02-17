#ifndef GEN_IMG_4_DENSE_H
#define GEN_IMG_4_DENSE_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "OpenGlInit.h"
#include "../utils/utils.hpp"
using namespace std;
using namespace cv;
class GenImg4Dense {
public:
	GenImg4Dense(int _width, int _height){
		width = _width;
		height = _height;
		//float amp = 0.05;
		//Mat camX(10 * 10, 1, CV_32FC1);
		//Mat camY(10 * 10, 1, CV_32FC1);
		//int ind = 0;
		//for (int i = -5; i < 5; i++){
		//	for (int j = -5; j < 5; j++){
		//		camX.at<float>(ind) = i*amp;
		//		camY.at<float>(ind) = j*amp;
		//		ind++;
		//	}
		//}
		float amp = 0.1;
		Mat camX = (Mat_<float>(9, 1) << 0, 1, -1, 0, 1, - 1, 0, 1, -1 );
		camX = camX*amp;
		Mat camY = (Mat_<float>(9, 1) << 0, 0, 1, 1, 1, - 1, -1, -1, 0 );
		camY = camY*amp;

		OpenGlInit glObj(2, _width, _height);
		for (int i = 0; i < 9; i++) {
			glObj.setCam(camX.at<float>(i), camY.at<float>(i), -3);
			glObj.renderOneFrame();
			float* imagP = glObj.clipScreen();
			char tempDigitStr[20];
			sprintf(tempDigitStr, "denseImg//den%02d", i);
			string fileName = tempDigitStr;
			Mat img1 = saveImgFromGL(imagP, fileName.c_str(), width, height);
		}
	}
private:
	int width;
	int height;
};

#endif
