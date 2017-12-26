#ifndef  FACEALL_H
#define  FACEALL_H
#include <stdio.h>
#include <string>
using namespace std;
namespace faceall{
typedef void* faceall_handle_t ;
class FaceallSuperIR{
    public:
    FaceallSuperIR(){};
    ~FaceallSuperIR(){};
    int load(faceall_handle_t handle, const std::string model_path); 
    int predict(faceall_handle_t handle, const string imagename,float* prediction);
    int loadAndPredict(const std::string model_path, const string imagename,const string outputimage);
};
void faceall_example();
}
#endif

