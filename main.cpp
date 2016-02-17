#include <iostream>

#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;
//using namespace cv::cuda;

int main(){
	Mat img = imread("loli.jpg");
	cout << CV_8UC4 << endl;
	imshow("First Image", img);
	cuda::GpuMat d_img(img);
	cuda::cvtColor(d_img, d_img,CV_BGR2RGBA);
	cout << d_img.type() << endl;
	cuda::GpuMat d_img_re;
	//GpuMat d_img_re(img.rows,img.cols, CV_8UC1, CvScalar(0, 0, 0, 0));
	cuda::meanShiftFiltering(d_img, d_img_re, 10, 10, TermCriteria(TermCriteria::MAX_ITER, 5, 1));
	Mat img_re;
	d_img_re.download(img_re);
	imshow("Re Image", img_re);
	waitKey(0);
	return 0;
}