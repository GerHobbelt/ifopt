#include <ifopt/problem.h>
#include <nlopt.h>

namespace Nlopt {
class NloptAdapter {
 public:
  using Problem  = ifopt::Problem;
  using VectorXd = Problem::VectorXd;
  using Jacobian = Problem::Jacobian;

  void Solve();

  explicit NloptAdapter(Problem& nlp, bool finite_diff = false);
  virtual ~NloptAdapter();

 private:
  Problem* nlp_;
  bool finite_diff_;
  nlopt_opt_s* opt_;

  static double eval_f(unsigned n, const double *x, double *grad_f, void *data);
  static double eval_g(unsigned n, const double *x, double *grad_g, void *data);
};
}  // namespace Nlopt
