#include "utility.h"

bool Utility::inBorder(const Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void Utility::reduceVector(vector<Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void Utility::reduceVector4Line(vector<Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i+=2)
        if (status[i] && status[i+1]){
            v[j++] = v[i];
            v[j++] = v[i+1];
        }
    v.resize(j);
}

void Utility::reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void Utility::DeleteCVMatRow( Mat &mat_in, unsigned int target_row){
  if ( target_row < 0 || target_row >= mat_in.rows )
    return;

  Mat upper_mat, lower_mat, mat_out;

  if ( target_row == 0 ){
    mat_in(Range(1, mat_in.rows), Range(0, mat_in.cols)).copyTo(upper_mat);
    mat_in = upper_mat;
  }
  else if ( target_row == mat_in.rows-1 ){
    mat_in(Range(0, mat_in.rows-1), Range(0, mat_in.cols)).copyTo(upper_mat);
    mat_in = upper_mat;
  }
  else{
    mat_in(Range(0, target_row), Range(0, mat_in.cols)).copyTo(upper_mat);    
    mat_in(Range(target_row+1, mat_in.rows), Range(0, mat_in.cols)).copyTo(lower_mat);

    vconcat(upper_mat, lower_mat, mat_out);
    mat_in = mat_out;
  }
}
