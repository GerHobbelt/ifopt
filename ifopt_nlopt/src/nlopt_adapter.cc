#include <ifopt/nlopt_adapter.h>
#include <iostream>

namespace Nlopt {

NloptAdapter::NloptAdapter(Problem& nlp, bool finite_diff)
{
  nlp_         = &nlp;
  finite_diff_ = finite_diff;
  opt_ = nlopt_create(NLOPT_LD_AUGLAG, nlp_->GetNumberOfOptimizationVariables());
}

void NloptAdapter::Solve()
{
  int major;
  int minor;
  int bugfix;
  nlopt_version(&major, &minor, &bugfix);
  std::cout << "Nlopt version: " << major << "." << minor << "." << bugfix
            << std::endl;
  nlopt_set_maxtime(opt_, 10);
  nlopt_set_ftol_rel(opt_, 1e-4);
  nlopt_set_ftol_abs(opt_, 1e-4);

  auto n          = nlp_->GetNumberOfOptimizationVariables();
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

  // double* g_l   = new double[n];
  // double* g_u   = new double[n];
  // auto bounds_g = nlp_->GetBoundsOnConstraints();
  // for (std::size_t c = 0; c < bounds_g.size(); ++c) {
  //   g_l[c] = bounds_g.at(c).lower_;
  //   g_u[c] = bounds_g.at(c).upper_;
  //
  //   if (g_l[c] == g_u[c]) {
  //     std::cout << "Equality constraint: " << c << std::endl;
  //     auto data     = std::make_shared<NloptAdapterConstraintWrapper>();
  //     data->adapter = this;
  //     data->index   = static_cast<int>(c);
  //
  //     auto eqRes = nlopt_add_equality_constraint(opt_, NloptAdapter::eval_g,
  //                                                &data, 1e-8);
  //     std::cout << "Add equality constraint result: " << eqRes << std::endl;
  //   } else if (std::isinf(g_l[c])) {
  //     std::cout << "Inequality constraint lower: " << c << std::endl;
  //   } else {
  //     std::cout << "Inequality constraint upper: " << c << std::endl;
  //   }
  // }

  auto constraints = nlp_->GetConstraints();
  auto vec         = constraints.GetComponents();
  for (size_t i = 0; i < vec.size(); i++) {
    auto constraint    = vec[i];
    auto bounds        = constraint->GetBounds();
    int numConstraints = constraint->GetRows();
    for (int j = 0; j < numConstraints; j++) {
      ifopt::Bounds bound = bounds[j];
      double lower        = bound.lower_;
      double upper        = bound.upper_;
      auto data           = NloptAdapterConstraintWrapper{*constraint, j};
      if (lower == upper) {
        data.equality = 0;
        data.bounds   = lower;
        std::cout << "Add equality constraint: " << std::endl;
        auto res = nlopt_add_equality_constraint(opt_, NloptAdapter::eval_g, &data, 1e-8);
        std::cout << "Add equality constraint result: " << res << std::endl;
      } else if (std::isinf(lower) && !std::isinf(upper)) {
        data.equality = 1;
        data.bounds   = upper;
        std::cout << "Add inequality constraint: " << std::endl;
        auto res= nlopt_add_inequality_constraint(opt_, NloptAdapter::eval_g, &data,
                                        1e-8);
        std::cout << "Add inequality constraint result: " << res << std::endl;
      } else {
        data.equality = 1;
        data.bounds   = lower;
        std::cout << "Add inequality constraint: " << std::endl;
        auto res = nlopt_add_inequality_constraint(opt_, NloptAdapter::eval_g, &data,
                                        1e-8);
        std::cout << "Add inequality constraint result: " << res << std::endl;
      }
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
  std::cout << "Optimal objective: " << minF << std::endl;

  std::cout << "Optimal x: " << x[0] << ", " << x[1] << std::endl;

  delete[] x;
  delete[] x_lower;
  delete[] x_upper;
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
  auto adapter = static_cast<NloptAdapterConstraintWrapper*>(data);
  Eigen::Map<const Eigen::VectorXd> inputX(x, n);
  auto gEig  = adapter->component.GetValues(inputX);
  auto jac   = adapter->component.GetJacobian(inputX);
  auto index = adapter->index;
  if (grad_g) {
    for (int i = 0; i < n; i++) {
      grad_g[i] = jac.coeff(index, i);
    }
  }

  // if (adapter->equality == 0)
  //   return std::abs(adapter->bounds - gEig[index]);

  return adapter->bounds - gEig[index];
}

}  // namespace Nlopt