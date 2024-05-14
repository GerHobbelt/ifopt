#include <iostream>

#include <ifopt/nlopt_adapter.h>
#include <ifopt/problem.h>
#include <ifopt/test_vars_constr_cost.h>

using namespace ifopt;

int main() {
  Problem nlp;
  nlp.AddVariableSet(std::make_shared<ExVariables>());
  nlp.AddConstraintSet(std::make_shared<ExConstraint>());
  nlp.AddCostSet(std::make_shared<ExCost>());
  nlp.PrintCurrent();

  Nlopt::NloptAdapter nlopt_adapter(nlp, false);
  nlopt_adapter.Solve();

  std::cout << "Test passed!" << std::endl;

}