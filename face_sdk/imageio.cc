#include "imageio.h"
#define scale_value 127.5
using namespace std;
using namespace tensorflow;
namespace faceall {
Status ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";
  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name),
                                               file_name);
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBicubic(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  //
    Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
     {input_std});
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));

  return Status::OK();
}

// Given an output tensor with 4d, reduce dim and output jpg image
Status SaveTensorToImageFile(const string& file_name, const Tensor* out_tensor) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    auto output_image_data = tensorflow::ops::Reshape(root, *out_tensor, { 64, 64, 3 });
    auto recover_data=Multiply(root, Add(root, output_image_data, {float(1.0)}),{float(127.5)});

    auto output_image_data_cast = tensorflow::ops::Cast(root, recover_data, tensorflow::DT_UINT8);
    auto output_image = tensorflow::ops::EncodeJpeg(root, output_image_data_cast);
    auto output_op = tensorflow::ops::WriteFile(root.WithOpName("sr_image"), file_name, output_image);
    string output_name = "sr_image";
    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    Status writeResult = session->Run({}, {}, { output_name }, {});
    std::cout<<"image save..."<<std::endl;
    return writeResult;
}
}



