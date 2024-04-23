#include "ortools_linprog.h"
#include "ortools/linear_solver/linear_solver.h"
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

void ortools_linprog::linprog(
    std::vector<float> &solution_values,
    const int &varNum,
    const int &srcNum,
    const int &tarNum,
    const cv::Mat weight,
    const cv::Mat &A,
    const cv::Mat &b,
    const float &beq
) {

  int n_vars = varNum + srcNum + tarNum;
  int n_cts = srcNum + tarNum;

  operations_research::MPSolver* const solver = operations_research::MPSolver::CreateSolver("GLOP");
  operations_research::MPVariable *vars[n_vars];
  for (int i = 0; i < n_vars; i++) {
    vars[i] = solver->MakeNumVar(0.0, operations_research::MPSolver::infinity(), "var" + std::to_string(i));
  }

  operations_research::MPConstraint *cts[n_cts + 1];
  for (int i = 0; i < n_cts; i++) {
    cts[i] = solver->MakeRowConstraint(
        -operations_research::MPSolver::infinity(), (double) b.at<float>(i, 0), "ct" + std::to_string(i)
    );
  }
  for (int i = 0; i < n_cts; i++) {
    for (int j = 0; j < n_vars; j++) {
      cts[i]->SetCoefficient(vars[j], (double) A.at<float>(i, j));
    }
  }
  cts[n_cts] = solver->MakeRowConstraint(beq, beq, "ct" + std::to_string(n_cts));
  for (int j = 0; j < n_vars; j++) {
    cts[n_cts]->SetCoefficient(vars[j], 1.0);
  }

  operations_research::MPObjective *const objective = solver->MutableObjective();
  for (int i = 0; i < n_vars; i++) {
    objective->SetCoefficient(vars[i], (double) weight.at<float>(0, i));
  }
  objective->SetMinimization();
  
  operations_research::MPSolverParameters parameters;
  
  parameters.SetIntegerParam(
            operations_research::MPSolverParameters::IntegerParam::LP_ALGORITHM,
            operations_research::MPSolverParameters::LpAlgorithmValues::DUAL);

  solver->Solve(parameters);

  for (int i = 0; i < n_vars; i++) {
    solution_values.push_back((float) vars[i]->solution_value());
  }
}