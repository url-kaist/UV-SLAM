#pragma once

#include "parameters.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class Utility{
public:
	bool inBorder(const Point2f &pt);
	void reduceVector(vector<Point2f> &v, vector<uchar> status);
	void reduceVector4Line(vector<Point2f> &v, vector<uchar> status);
	void reduceVector(vector<int> &v, vector<uchar> status);
	void DeleteCVMatRow( Mat &mat_in, unsigned int target_row);
};