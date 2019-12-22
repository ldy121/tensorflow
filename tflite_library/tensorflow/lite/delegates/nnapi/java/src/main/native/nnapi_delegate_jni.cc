/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <jni.h>

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

#include <vector> // by LDY
#include <string> // by LDY

#include <android/log.h>
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "LDY_nnapi_delegate_jni", __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG  , "LDY_nnapi_delegate_jni", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO   , "LDY_nnapi_delegate_jni", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN   , "LDY_nnapi_delegate_jni", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , "LDY_nnapi_delegate_jni", __VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_createDelegate(JNIEnv* env,
                                                            jclass clazz) {
  return reinterpret_cast<jlong>(tflite::NnApiDelegate());
}

// by LDY
JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_createDelegateOpt(JNIEnv* env,
                                                            jclass clazz,
							    jstring acceleratorName,
							    jstring executionPreference) {
  tflite::StatefulNnApiDelegate::Options options;
  const char * accelerator = NULL;
  const char * preference = NULL;
  jlong ret;

  if (executionPreference != NULL) {
    preference = env->GetStringUTFChars(executionPreference, nullptr); 
    if (preference != nullptr) {
      std::string string_execution_preference = std::string(preference);
      if (!string_execution_preference.empty()) {
        tflite::StatefulNnApiDelegate::Options::ExecutionPreference
            execution_preference =
                tflite::StatefulNnApiDelegate::Options::kUndefined;
        if (string_execution_preference == "low_power") {
          execution_preference =
              tflite::StatefulNnApiDelegate::Options::kLowPower;
        } else if (string_execution_preference == "sustained_speed") {
          execution_preference =
              tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
        } else if (string_execution_preference == "fast_single_answer") {
          execution_preference =
              tflite::StatefulNnApiDelegate::Options::kFastSingleAnswer;
        } else if (string_execution_preference == "undefined") {
          execution_preference =
              tflite::StatefulNnApiDelegate::Options::kUndefined;
        } else {
          LOGE(string_execution_preference.c_str());
        }   
        options.execution_preference = execution_preference;
      }
    }
  }

  if (acceleratorName != NULL) {
    accelerator = env->GetStringUTFChars(acceleratorName, nullptr);
    if (accelerator != nullptr) {
      std::string accelerator_name = std::string(accelerator);
      options.accelerator_name = accelerator_name.c_str();
    }
  }

  ret = reinterpret_cast<jlong>(tflite::NnApiDelegate(options));
  
  if (accelerator != NULL) {
    env->ReleaseStringUTFChars(acceleratorName, accelerator);
  }

  if (preference != NULL) {
    env->ReleaseStringUTFChars(executionPreference, preference);
  }

  return ret;
}

tflite::StatefulNnApiDelegate* convertLongToNnApiDelegate(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    return nullptr;
  }
  return reinterpret_cast<tflite::StatefulNnApiDelegate*>(handle);
}

// by LDY
JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_getDeviceNames(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  tflite::StatefulNnApiDelegate* delegate = convertLongToNnApiDelegate(env, handle);
  if (delegate == nullptr) return nullptr;

  std::vector<std::string*> * devices = delegate->getDeviceNames();

  if (devices != NULL && devices->size() > 0) {
    jclass string_class = env->FindClass("java/lang/String");
    if (string_class == nullptr) {
      return nullptr;
    }
    jobjectArray names = static_cast<jobjectArray>(
      env->NewObjectArray(devices->size(), string_class, env->NewStringUTF("")));
    for (int i = 0; i < devices->size(); ++i) {
      env->SetObjectArrayElement(names, i,
                               env->NewStringUTF((*devices)[i]->c_str()));
    }
    
    return names;
  } else {
    return nullptr;
  }
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_getHWexeTime(JNIEnv* env,
                                                         jclass clazz,
							 jlong handle) { // by LDY
  tflite::StatefulNnApiDelegate* delegate = convertLongToNnApiDelegate(env, handle);
  if (delegate == nullptr) return 0;

  return delegate->get_hw_exe_time();
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_getDriverExeTime(JNIEnv* env,
                                                            jclass clazz,
							    jlong handle) { // by LDY
  tflite::StatefulNnApiDelegate* delegate = convertLongToNnApiDelegate(env, handle);
  if (delegate == nullptr) return 0;

  return delegate->get_driver_exe_time();
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
