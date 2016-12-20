#include "wrapper.h"

FANN_Wrapper::FANN_Wrapper(FANN::neural_net* net) {
  this->net = net;
}

int FANN_Wrapper::getNumberInputs() {
  return (int) this->getNeuralNet()->get_num_input();
}

int FANN_Wrapper::getNumberOutputs() {
  return (int) this->getNeuralNet()->get_num_output();
}

FANN::neural_net* FANN_Wrapper::getNeuralNet() {
  return this->net;
}
std::vector<float> FANN_Wrapper::run(std::vector<float> input) {
  float inputA[this->getNumberInputs()];
  for (int i = 0; i < this->getNumberInputs(); i++) {
    inputA[i] = input[i];
  }

  float* output = this->getNeuralNet()->run(inputA);

  std::vector<float> outputV;
  for (int i = 0; i < this->getNumberOutputs(); i++) {
    outputV.push_back(output[i]);
  }

  return outputV;
}
