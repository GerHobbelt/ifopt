#include <iostream>

#include <ifopt/nlopt_adapter.h>
#include <ifopt/problem.h>
#include <ifopt/test_vars_constr_cost.h>

using namespace ifopt;

int main()
{
  std::cout << "Test ifopt with nlopt" << std::endl;
  Problem nlp;
  nlp.AddVariableSet(std::make_shared<ExVariables>());
  nlp.AddConstraintSet(std::make_shared<ExConstraint>());
  nlp.AddCostSet(std::make_shared<ExCost>());
  nlp.PrintCurrent();

  auto cost = nlp.GetCosts();
  std::cout << cost.GetValues() << std::endl;

  Problem::VectorXd cntVars(2);
  cntVars << 1, 1;

  Nlopt::NloptAdapter nlopt_adapter(nlp, false);
  nlopt_adapter.Solve();

  std::cout << "Test passed!" << std::endl;
}