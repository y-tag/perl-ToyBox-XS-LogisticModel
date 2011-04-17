#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#ifdef __cplusplus
}
#endif

#include "ppport.h"
#include <lbfgs.h>
#include <string>
#include <vector>
#include <map>
#include <cfloat>
#include <cmath>

typedef std::map<int, int> IntToIntMap;
typedef std::map<int, double> IntToDoubleMap;
typedef std::map<std::string, int> StrToIntMap;
typedef std::map<std::string, double> StrToDoubleMap;
typedef std::map<std::string, std::map<std::string, double> > Str2ToDoubleMap;

typedef std::vector<IntToDoubleMap> IDMapVector;

typedef struct{
  IDMapVector data;
  std::vector<int> labels;
  int fnum;
  int lnum;
} instance_struct;

class LogisticModel{
  public:
    LogisticModel();
    ~LogisticModel();
    void AddInstance(const StrToIntMap &doc, const std::string &label);
    void Train(const int max_iterations, const int max_linesearch);
    StrToDoubleMap Predict(const StrToIntMap &doc);
  private:
    int numData;
    StrToIntMap fdict;
    StrToIntMap ldict;
    IDMapVector data;
    std::vector<int> labels;
    std::vector<double> weight;
};

static lbfgsfloatval_t evaluate(
  void *instance,
  const lbfgsfloatval_t *x,
  lbfgsfloatval_t *g,
  const int n,
  const lbfgsfloatval_t step
);
static int progress(
  void *instance,
  const lbfgsfloatval_t *x,
  const lbfgsfloatval_t *g,
  const lbfgsfloatval_t fx,
  const lbfgsfloatval_t xnorm,
  const lbfgsfloatval_t gnorm,
  const lbfgsfloatval_t step,
  int n,
  int k,
  int ls
);


LogisticModel::LogisticModel()
  : numData(0), fdict(), data(), labels(), weight()
{
}

LogisticModel::~LogisticModel()
{
}

void
LogisticModel::AddInstance(const StrToIntMap &doc, const std::string &label)
{

  IntToDoubleMap int_doc;

  for (StrToIntMap::const_iterator it = doc.begin(); it != doc.end(); ++it) {
    if (fdict.find(it->first) == fdict.end()) {
      int fnum = fdict.size();
      fdict[it->first] = fnum;
    }
    int feature = fdict[it->first];
    int_doc[feature] = it->second;
  }

  if (ldict.find(label) == ldict.end()) {
    int lnum = ldict.size();
    ldict[label] = lnum;
  }

  data.push_back(int_doc);
  labels.push_back(ldict[label]);
  ++numData;

  return;
}

void
LogisticModel::Train(const int max_iterations, const int max_linesearch)
{
  instance_struct instance;
  instance.data   = data;
  instance.labels = labels;
  instance.fnum   = fdict.size();
  instance.lnum   = ldict.size();

  int weight_size = fdict.size() * (ldict.size() - 1);

  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(weight_size);

  if (x == NULL) {
    fprintf(stderr, "lbfgs_malloc error\n");
    return;
  }
  
  for (int i = 0; i < weight_size; ++i) {
    x[i] = 0.0;
  }

  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);

  param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
  param.max_iterations = max_iterations;
  param.max_linesearch = max_linesearch;

  int ret = lbfgs(weight_size, x, &fx, evaluate, progress, &instance, &param);

  fprintf(stderr, "L-BFGS optimization terminated with status code = %d\n", ret);

  for (int i = 0; i < weight_size; ++i) {
    weight.push_back(x[i]);
  }

  lbfgs_free(x);
  x = NULL;

  return;
}

StrToDoubleMap
LogisticModel::Predict(const StrToIntMap &doc)
{

  StrToDoubleMap result;
  std::vector<double> exp_sum;
  double denom = 0.0;

  int fnum = fdict.size();
  int lnum = ldict.size();

  // for 1, ..., L-1
  for (int i = 0; i < lnum - 1; ++i) {
    double tmp_sum = 0.0;

    for (StrToIntMap::const_iterator fit = doc.begin(); fit != doc.end(); ++fit) {
      if (fdict.find(fit->first) == fdict.end()) {
        continue;
      }
      int feature = fnum * i + fdict[fit->first];
      tmp_sum += weight[feature] * fit->second;
    }
    tmp_sum = exp(tmp_sum);
    denom += tmp_sum;
    exp_sum.push_back(tmp_sum);
  }

  // for L
  denom += 1.0;
  exp_sum.push_back(1.0);

  for (StrToIntMap::iterator lit = ldict.begin(); lit != ldict.end(); ++lit) {
    result[lit->first] = exp_sum[lit->second] / denom;
  }

  return result;
}

static lbfgsfloatval_t evaluate(
  void *instance,
  const lbfgsfloatval_t *x,
  lbfgsfloatval_t *g,
  const int n,
  const lbfgsfloatval_t step
)
{
  instance_struct *par = static_cast<instance_struct*>(instance);
  IDMapVector data = par->data;
  std::vector<int> labels = par->labels;
  int fnum = par->fnum;
  int lnum = par->lnum;

  lbfgsfloatval_t fx = 0.0;
  for (int i = 0; i < n; ++i) {
    g[i] = 0.0;
  }

  for (int i = 0; i < data.size(); ++i) {
    IntToDoubleMap datum = data[i];
    int label = labels[i];

    std::vector<double> numer;
    double denom = 0.0;

    for (int j = 0; j < lnum - 1; ++j) {
      double linear_sum = 0.0;

      for (IntToDoubleMap::const_iterator it = datum.begin(); it != datum.end(); ++it) {
        int feature = fnum * j + it->first;
        linear_sum += x[feature] * it->second;
      }

      double exp_sum = exp(linear_sum);
      numer.push_back(exp_sum);
      denom += exp_sum;

      if (label == j) {
        fx -= linear_sum;
      }
    }

    denom += 1.0;

    for (int j = 0; j < lnum - 1; ++j) {
      double y = (label == j) ? 1.0 : 0.0;
      double diff = y - (numer[j] / denom);

      for (IntToDoubleMap::const_iterator it = datum.begin(); it != datum.end(); ++it) {
        int feature = fnum * j + it->first;
        g[feature] -= diff * it->second;
      }
    }

    fx += log(denom);
  }

  return fx;
}

static int progress(
  void *instance,
  const lbfgsfloatval_t *x,
  const lbfgsfloatval_t *g,
  const lbfgsfloatval_t fx,
  const lbfgsfloatval_t xnorm,
  const lbfgsfloatval_t gnorm,
  const lbfgsfloatval_t step,
  int n,
  int k,
  int ls
)
{
  //fprintf(stderr, "%5d fx: %10.16f, xnorm: %f, gnorm: %f\n", k, fx, xnorm, gnorm);
  fprintf(stderr, "%5d %10.16f\n", k, fx);
  return 0;
}


MODULE = ToyBox::XS::LogisticModel		PACKAGE = ToyBox::XS::LogisticModel	

LogisticModel *
LogisticModel::new()

void
LogisticModel::DESTROY()

void
LogisticModel::xs_add_instance(attributes_input, label_input)
  SV * attributes_input
  char* label_input
CODE:
  {
    HV *hv_attributes = (HV*) SvRV(attributes_input);
    SV *val;
    char *key;
    I32 retlen;
    int num = hv_iterinit(hv_attributes);
    std::string label = std::string(label_input);
    StrToIntMap attributes;

    for (int i = 0; i < num; ++i) {
      val = hv_iternextsv(hv_attributes, &key, &retlen);
      attributes[key] = (int)SvIV(val);
    }

    THIS->AddInstance(attributes, label);
  }

void
LogisticModel::xs_train(max_iterations, max_linesearch)
  int max_iterations
  int max_linesearch
CODE:
  {
    THIS->Train(max_iterations, max_linesearch);
  }

SV*
LogisticModel::xs_predict(attributes_input)
  SV * attributes_input
CODE:
  {
    HV *hv_attributes = (HV*) SvRV(attributes_input);
    SV *val;
    char *key;
    I32 retlen;
    int num = hv_iterinit(hv_attributes);
    StrToIntMap attributes;
    StrToDoubleMap result;

    for (int i = 0; i < num; ++i) {
      val = hv_iternextsv(hv_attributes, &key, &retlen);
      attributes[key] = (int)SvIV(val);
    }

    result = THIS->Predict(attributes);

    HV *hv_result = newHV();
    for (StrToDoubleMap::iterator it = result.begin(); it != result.end(); ++it) {
      const char *const_key = (it->first).c_str();
      SV* val = newSVnv(it->second);
      hv_store(hv_result, const_key, strlen(const_key), val, 0); 
    }

    RETVAL = newRV_inc((SV*) hv_result);
  }
OUTPUT:
  RETVAL
  
