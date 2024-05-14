#include <ifopt/nlopt_adapter.h>
#include <iostream>

namespace Nlopt {

NloptAdapter::NloptAdapter(Nlopt::NloptAdapter::Problem& nlp, bool finite_diff)
{
  nlp_         = &nlp;
  finite_diff_ = finite_diff;
}

void NloptAdapter::Solve()
{
  int major;
  int minor;
  int bugfix;
  nlopt_version(&major, &minor, &bugfix);
  std::cout << "Nlopt version: " << major << "." << minor << "." << bugfix
            << std::endl;

  auto n = nlp_->GetNumberOfOptimizationVariables();
  std::cout << "Number of optimization variables: " << n << std::endl;
  opt_ = nlopt_create(NLOPT_LD_SLSQP, n);

  auto bounds_x   = nlp_->GetBoundsOnOptimizationVariables();
  double* x_lower = new double[n];
  double* x_upper = new double[n];
  for (std::size_t c = 0; c < bounds_x.size(); ++c) {
    x_lower[c] = bounds_x.at(c).lower_;
    x_upper[c] = bounds_x.at(c).upper_;
    std::cout << "Variable " << c << " lower bound: " << x_lower[c]
              << ", upper bound: " << x_upper[c] << std::endl;
  }

  auto setLower = nlopt_set_lower_bounds(opt_, x_lower);
  std::cout << "Set lower bounds result: " << setLower << std::endl;
  auto setUpper = nlopt_set_upper_bounds(opt_, x_upper);
  std::cout << "Set upper bounds result: " << setUpper << std::endl;

  double* g_l   = new double[n];
  double* g_u   = new double[n];
  auto bounds_g = nlp_->GetBoundsOnConstraints();
  for (std::size_t c = 0; c < bounds_g.size(); ++c) {
    g_l[c] = bounds_g.at(c).lower_;
    g_u[c] = bounds_g.at(c).upper_;

    if (g_l[c] == g_u[c]) {
      std::cout << "Equality constraint: " << c << std::endl;
      auto eqRes =
          nlopt_add_equality_constraint(opt_, NloptAdapter::eval_g, this, 1e-8);
      std::cout << "Add equality constraint result: " << eqRes << std::endl;
    } else {
      std::cout << "Inequality constraint: " << c << std::endl;
    }
  }

  nlopt_result setRes =
      nlopt_set_min_objective(opt_, NloptAdapter::eval_f, this);
  std::cout << "Set min objective result: " << setRes << std::endl;

  double* x                                 = new double[n];
  VectorXd x_all                            = nlp_->GetVariableValues();
  Eigen::Map<VectorXd>(&x[0], x_all.rows()) = x_all;
  std::cout << "Initial x: " << x[0] << ", " << x[1] << std::endl;

  double minF;
  auto res = (nlopt_result)nlopt_optimize(opt_, x, &minF);
  std::cout << "Nlopt result: " << res << std::endl;
  std::cout << "Optimal f: " << minF << std::endl;

  std::cout << "Optimal x: " << x[0] << ", " << x[1] << std::endl;

  delete[] x;
  delete[] x_lower;
  delete[] x_upper;
  delete[] g_l;
  delete[] g_u;
}

NloptAdapter::~NloptAdapter()
{
  std::cout << "Destroying NloptAdapter" << std::endl;
  nlopt_destroy(opt_);
}

double count = 0;
double NloptAdapter::eval_f(unsigned n, const double* x, double* grad_f,
                            void* data)
{
  auto adapter   = static_cast<NloptAdapter*>(data);
  auto obj_value = adapter->nlp_->EvaluateCostFunction(x);
  if (grad_f) {
    Eigen::VectorXd grad =
        adapter->nlp_->EvaluateCostFunctionGradient(x, adapter->finite_diff_);
    Eigen::Map<Eigen::MatrixXd>(grad_f, n, 1) = grad;
  }
  return obj_value;
}

double NloptAdapter::eval_g(unsigned n, const double* x, double* grad_g,
                            void* data)
{
  std::cout << "num constraints: " << n << std::endl;
  auto adapter = static_cast<NloptAdapter*>(data);
  auto gEig    = adapter->nlp_->EvaluateConstraints(x);
  auto jac     = adapter->nlp_->GetJacobianOfConstraints();
  std::cout << "Jacobian: " << jac << std::endl;
  if (grad_g) {
    grad_g[0] = jac.coeff(0, 0);
    grad_g[1] = jac.coeff(0, 1);
  }

  // adapter->nlp_->EvalNonzerosOfJacobian(x, grad_g);
  // std::cout << "Equality: " << gEig[0] << std::endl;
  // std::cout << "grad_g: " << grad_g[0] << std::endl;
  return gEig[0] - 1;
}

}  // namespace Nlopt