#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h> 

#include "convertAhandaPovRayToStandard.h"

#include "utils/utils.hpp"
#include "chamo/headers.h"
#include "chamo/costVol_chamo.h"
#include "drawBall/GenImg4Dense.h"
#include "chamo/imgInputer.h"
using namespace cv;
using namespace std;

int App_main( int argc, char** argv )
{
    int numImg=5;
	int layers = 32;
	int imagesPerCV = 5;
	//imgInputer *images = &imgInputerChamo();
	imgInputer *images = &imgInputerAhanda();
	images->import(numImg);
    
	costVol_chamo cv(images->imgList[0], layers, 0.01, 0.0, images->projMats[0], images->intrinMat);
	float weightPerImg = 1.0 / imagesPerCV;
	for (int i = 1; i < imagesPerCV; i++){
		cv.updataCV(images->imgList[i], images->projMats[i], weightPerImg);
		//cout << i << endl;
		//cout << images->projMats[i] << endl;
		//showMat(cv.bottonVal);
	}
	showMat(cv.bottonVal);
	cv.getGradientImg();
	showMat(cv.g);
	do{
		for (int i = 0; i < 10; i++){
			cv.updataD();
		}
		showMat(cv.d);
		cv.updataA();
		showMat(cv.a);
	} while (true);
	
    return 0;
}

void genBallImag(){
	GenImg4Dense objGen(640,480);
}

int main(int argc, char** argv){
	App_main(argc, argv);
	//efficience_try();
	//efficience_try_tex();
	//genBallImag();
	return 0;
}


