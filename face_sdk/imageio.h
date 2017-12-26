#ifndef  FACEALL_IMAGEIO_H
#define  FACEALL_IMAGEIO_H
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
namespace faceall {
using namespace tensorflow;
Status ReadTensorFromImageFile(std::string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors);
Status SaveTensorToImageFile(const std::string& file_name, const Tensor* out_tensor) ;
}
#endif