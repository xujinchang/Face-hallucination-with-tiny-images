#include "faceall.h"
using namespace faceall;
int main(){
    faceall_example();
    FaceallSuperIR faceall;
    faceall_handle_t handle;
    std::string model_path="nn.pb";
    faceall.loadAndPredict(model_path,"ori.jpg","output.jpg");
    return 0;
}

