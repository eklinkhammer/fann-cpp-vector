#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "wrapper.h"
#include <iostream>

const std::string configurationFile = "/Users/klinkhae/research/cpp_ws/fann-analysis/input/agent.policy";

class WrapperTest : public::testing::Test {
public:
  FANN::neural_net* net = new FANN::neural_net(configurationFile);
  FANN_Wrapper* wrapper = new FANN_Wrapper(net);
};

TEST_F(WrapperTest, testRunSingle) {
  std::vector<float> input = {0,0,0,0,0.1,0,0,0};
  std::vector<float> expected = {0.978634,0.89534855};
  std::vector<float> output = this->wrapper->run(input);

  for (int i = 0; i < 2; i++) {
    EXPECT_FLOAT_EQ(expected[i], output[i]);
  }
}

TEST_F(WrapperTest, testRunMultipleCopies) {
  std::vector<float> input = {0,0,0,0,0.1,0,0,0};
  std::vector<float> expected = {0.978634,0.89534855};

  this->wrapper->run(input);
  this->wrapper->run(input);
  
  std::vector<float> output = this->wrapper->run(input);

  for (int i = 0; i < 2; i++) {
    EXPECT_FLOAT_EQ(expected[i], output[i]);
  }
}

TEST_F(WrapperTest, testRunMultipleNoBuffer) {
  std::vector<float> input = {0,0,0,0,0.1,0,0,0};
  std::vector<float> expected = {0.978634,0.89534855};
  std::vector<float> output = this->wrapper->run(input);

  input[0] = 1;
  this->wrapper->run(input);

  for (int i = 0; i < 2; i++) {
    EXPECT_FLOAT_EQ(expected[i], output[i]);
  }
}

TEST_F(WrapperTest, testTrain) {

  FANN::neural_net* net = new FANN::neural_net(configurationFile);
  FANN_Wrapper* newWrapper = new FANN_Wrapper(net);

  std::vector<float> input = {1,0,-1,0,0.1,0,0,0};
  std::vector<float> output = {0.55,0.72};
  newWrapper->train(input,output);
  
  std::vector<float> inputAfter = {0,0,0,0,0.1,0,0,0};
  std::vector<float> expected = {0.97832328,0.88256174};
  std::vector<float> outputAfter = newWrapper->run(inputAfter);
  
  for (int i = 0; i < 2; i++) {
    EXPECT_FLOAT_EQ(expected[i], outputAfter[i]);
  }
}

TEST_F(WrapperTest, testTrainInputTooSmall) {
  std::vector<float> input = {1,0,-1,0,0.1,0,0};
  std::vector<float> output = {0.55,0.72};
  int result = this->wrapper->train(input,output);

  EXPECT_EQ(0, result);
}

TEST_F(WrapperTest, testTrainOutputTooSmall) {
  std::vector<float> input = {1,0,-1,0,0.1,0,0,0};
  std::vector<float> output = {0.55};
  int result = this->wrapper->train(input,output);

  EXPECT_EQ(0, result);
}

TEST_F(WrapperTest, testTrainInputCorrectSize) {
  std::vector<float> input = {1,0,-1,0,0.1,0,0,0};
  std::vector<float> output = {0.55,0.72};
  int result = this->wrapper->train(input,output);

  EXPECT_EQ(1, result);
}
