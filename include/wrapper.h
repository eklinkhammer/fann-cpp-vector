#include <vector>
#include <fann.h>
#include <fann_cpp.h>

class FANN_Wrapper {
 public:
  FANN_Wrapper(FANN::neural_net*);
  int getNumberInputs();
  int getNumberOutputs();
  int train(std::vector<float>,std::vector<float>);
  std::vector<float> run(std::vector<float>);
  FANN::neural_net* getNeuralNet();
 private:
  FANN::neural_net* net;
};
