load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
cc_library(
      name = "faceall",
      srcs = ["faceall.cc","imageio.cc"],
      hdrs= ["faceall.h","imageio.h"],
      deps = [
                  "//tensorflow/cc:cc_ops",
                  "//tensorflow/cc:client_session",
                  "//tensorflow/core:tensorflow",
                  "//tensorflow/cc:scope",
              ],
          alwayslink = 1,
)


cc_binary(
        name = "test_faceall",
        srcs = ["test_faceall.cc"],
        deps = [
                    "//tensorflow/cc/faceall:faceall",
                ],
)
