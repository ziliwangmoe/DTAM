#ifndef IMG_INPUTER_H
#define IMG_INPUTER_H
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../utils/utils.hpp"
using namespace std;
using namespace cv;

class imgInputer{
protected:
	string dir;
public:
	int width, height;
	vector<Mat> imgList;
	vector<Mat> projMats;
	Mat intrinMat;
	int imgCount;
	virtual void import(int count) = 0;
};

class imgInputerChamo : public imgInputer{
	void import(int count);
};

class imgInputerAhanda : public imgInputer{
	void import(int count);
};

#endif