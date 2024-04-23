#ifndef ORTOOLS_LINPROG_H_
#define ORTOOLS_LINPROG_H_

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ortools_linprog {
  void linprog(
      std::vector<float> &solution_values,
      const int &varNum,
      const int &srcNum,
      const int &tarNum,
      const cv::Mat weight,
      const cv::Mat &A,
      const cv::Mat &b,
      const float &beq
  );
}

#endif