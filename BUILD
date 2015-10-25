cc_library(
  name = "histogram",
  srcs = ["histogram.cc"],
  hdrs = ["histogram.h"],
)

cc_library(
  name = "framework",
  srcs = ["framework.cc"],
  hdrs = ["framework.h"],
)

cc_library(
  name = "test",
  srcs = ["test.cc"],
  hdrs = ["test.h"],
)

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    ":framework",
    ":histogram",
    ":test",
  ],
)
