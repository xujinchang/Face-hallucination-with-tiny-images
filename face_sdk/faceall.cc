#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "faceall.h"
#include "imageio.h"
using namespace tensorflow;

#define INPUT_H 16
#define INPUT_W 16
#define  FACEALL_H
static float mean_value=127.5;
static float scale_value=127.5;
namespace faceall {
void faceall_example() {
  using namespace tensorflow::ops;
  Scope root = Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
  // Vector b = [3 5]
  auto b = Const(root, { {3.f, 5.f} });
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
}

int FaceallSuperIR::load(faceall_handle_t handle, 
                  const std::string model_path) {
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 0;
    }
    //Read the pb file into the grapgdef member
    tensorflow::GraphDef graphdef;
    tensorflow::Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
    if (!status_load.ok()) {
        std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
        std::cout << status_load.ToString() << "\n";
        return -1;
    }

    // Add the graph to the session
    tensorflow::Status status_create = session->Create(graphdef);
    if (!status_create.ok()) {
        std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
        return -1;
    }
    handle=(void*)session;
    return 0;
}

int FaceallSuperIR::predict(faceall_handle_t handle,
                     std::string imagename,
                     float* prediction) {
    tensorflow::Session* session=(tensorflow::Session*)handle;
    Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_H,INPUT_W,3})); 
    std::vector<Tensor> image_tensor_vec;
    image_tensor_vec.push_back(x);
    if(!ReadTensorFromImageFile(imagename,INPUT_H,INPUT_W,mean_value,scale_value,&image_tensor_vec).ok()){
        std::cout<<"ERROR: Loading image fail\n";
    };
    string input_name="WGAN-GP_model_16_64_ps_noise_near_conv_l2_msc/Placeholder_1";
    string output_name="WGAN-GP_model_16_64_ps_noise_near_conv_l2_msc/generator/Conv/Tanh";
    std::vector<std::pair<string, Tensor> > inputs;
    inputs.push_back(std::pair<string,Tensor>(input_name,image_tensor_vec[0]));

    std::vector<tensorflow::Tensor> outputs;
    std::cout<<"before run;"<<std::endl;
    tensorflow::Status status = session->Run(inputs, {output_name}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
        return -1;
    }

    return 0;
}

int FaceallSuperIR::loadAndPredict(const std::string model_path,
                     std::string imagename,
                     std::string outputimage) {
    std::unique_ptr<tensorflow::Session> session(NewSession(SessionOptions()));
    tensorflow::GraphDef graphdef;
    tensorflow::Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
    if (!status_load.ok()) {
        std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
        std::cout << status_load.ToString() << "\n";
        return -1;
    }

    // Add the graph to the session
    tensorflow::Status status_create = session->Create(graphdef);
    if (!status_create.ok()) {
        std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
        return -1;
    }                     
    std::vector<Tensor> image_tensor_vec;
    if(!ReadTensorFromImageFile(imagename,INPUT_H,INPUT_W,mean_value,scale_value,&image_tensor_vec).ok()){
        std::cout<<"ERROR: Loading image fail\n";
    }
    string input_name="WGAN-GP_model_16_64_ps_noise_near_conv_l2_msc/Placeholder_1:0";
    string output_name="WGAN-GP_model_16_64_ps_noise_near_conv_l2_msc/generator/Conv/Tanh";
    std::vector<std::pair<string, Tensor> > inputs;
    inputs.push_back(std::pair<string,Tensor>(input_name,image_tensor_vec[0]));
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status status_run = session->Run(inputs, {output_name}, {}, &outputs);
    if (!status_run.ok()) {
        std::cout << "ERROR: prediction failed..." << status_run.ToString() << std::endl;
        return -1;
    }
    std::cout<<"save success? "<<SaveTensorToImageFile(outputimage,&outputs[0]);
    return 0;
}
}