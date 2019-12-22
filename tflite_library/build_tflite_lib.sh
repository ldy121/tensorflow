#!/bin/sh

bazel build --cxxopt='-std=c++11' -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a tensorflow/lite/java:tensorflow-lite
