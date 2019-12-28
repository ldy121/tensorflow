/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package inference;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;

/**
 * Classifies images with Tensorflow Lite.
 */
public abstract class Inference {
  // Display preferences
  private static final float GOOD_PROB_THRESHOLD = 0.3f;
  private static final int SMALL_COLOR = 0xffddaa88;

  /** Tag for the {@link Log}. */
  private static final String TAG = "aplab evalution";

  /** Number of results to show in the UI. */
  private static final int RESULTS_TO_SHOW = 3;


  private static final int DIM_PIXEL_SIZE = 3;

  /** Preallocated buffers for storing image data in. */
  private int[] intValues = new int[getImageSizeX() * getImageSizeY()];

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel = null;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;

  /** Labels corresponding to the output of the vision model. */
  protected List<String> labelList;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  protected ByteBuffer imgData = null;

  /** multi-stage low pass filter * */
  private float[][] filterLabelProbArray = null;

  private static final int FILTER_STAGES = 3;
  private static final float FILTER_FACTOR = 0.4f;

  private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
      new PriorityQueue<>(
          RESULTS_TO_SHOW,
          new Comparator<Map.Entry<String, Float>>() {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
              return (o1.getValue()).compareTo(o2.getValue());
            }
          });

  /** holds a gpu delegate */
  Delegate gpuDelegate = null;

  /** Initializes an {@code ImageClassifier}. */
  protected void load_model(Activity activity) throws IOException {
    tfliteModel = loadModelFile(activity);
    tflite = new Interpreter(tfliteModel, tfliteOptions);
    labelList = loadLabelList(activity);
    imgData =
        ByteBuffer.allocateDirect(
            getBatchSize()
                * getImageSizeX()
                * getImageSizeY()
                * DIM_PIXEL_SIZE
                * getNumBytesPerChannel());
    imgData.order(ByteOrder.nativeOrder());
    filterLabelProbArray = new float[FILTER_STAGES][getNumLabels()];
  }

  private void recreateInterpreter() {
    if (tflite != null) {
      tflite.close();
      // TODO(b/120679982)
      // gpuDelegate.close();
      tflite = new Interpreter(tfliteModel, tfliteOptions);
    }
  }

  public void use16bitNNAPI() { // by LDY
    tfliteOptions.setUseNNAPI(true);
    tfliteOptions.setAllowFp16PrecisionForFp32(true);
    recreateInterpreter();
  }

  public void useNNAPI() {
    tfliteOptions.setUseNNAPI(true);
    recreateInterpreter();
  }

  public void setNumThreads(int numThreads) {
    tfliteOptions.setNumThreads(numThreads);
    recreateInterpreter();
  }

  public void setAccelerator(String accelerator) {
    tfliteOptions.setAccelerator(accelerator);
    recreateInterpreter();
  }

  public void setExecutionPreference(String executionPreference) {
    tfliteOptions.setOperationMode(executionPreference);
    recreateInterpreter();
  }

  /** Closes tflite to release resources. */
  public void close() {
    tflite.close();
    tflite = null;
    tfliteModel = null;
  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity) throws IOException {
    if (getLabelPath() == null) {
      return null;
    }

    List<String> labelList = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
    String line;
    while ((line = reader.readLine()) != null) {
      labelList.add(line);
    }
    reader.close();
    return labelList;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  public void generate_input(int zero_ratio) {
    Boolean[] zero_psition = new Boolean[getBatchSize() * getImageSizeX() * getImageSizeY()];
    Random rand = new Random();

    for (int i = 0; i < zero_psition.length; ++i) {
      zero_psition[i] = Boolean.FALSE;
    }

    rand.setSeed(0);
    for (int num_of_zero = zero_psition.length * zero_ratio / 100;
         num_of_zero > 0; --num_of_zero) {
      int val;
      for (val = rand.nextInt(zero_psition.length);
           (zero_psition[val]);
           val = rand.nextInt(zero_psition.length));
      zero_psition[val] = Boolean.TRUE;
    }

    for (int i = 0; i < zero_psition.length; ++i) {
      if (zero_psition[i] == Boolean.FALSE) {
        int val = ((rand.nextInt(254) + 1) << 16) |
                ((rand.nextInt(254) + 1) << 8) |
                ((rand.nextInt(254) + 1) << 0);
        addPixelValue(val);
      } else {
        addPixelValue(0);
      }
    }
  }

  private String time_metric = "default time";

  public String get_perf_metric() {
      return time_metric;
  }

  private int m_duration = 1;
  private Boolean m_hw_exe_time = false;

  public void set_hw_exe_time(Boolean b) {
    m_hw_exe_time = b;
  }

  public void set_duration(int sec) {
    m_duration = sec;
  }

  public float single_instance_evaluate_performance() {
    long time = 0;
    long duration = m_duration * 1000;
    long start_time;
    long latency = 0;
    long exe_time;
    int num_of_image = 0;

    for (latency = 0, time = 0, start_time = System.currentTimeMillis();
         time < duration; num_of_image += getBatchSize(), latency += exe_time) {
      runInference();
      time = System.currentTimeMillis() - start_time;

      exe_time = tflite.get_hw_time() / 1000000;
      if ((m_hw_exe_time) && (exe_time > 0 && exe_time < 1000000000)) {
        time_metric = "NN H/W latency ";
        continue;
      }

      exe_time = tflite.getLastNativeInferenceDurationNanoseconds() / 1000000;
      time_metric = "App Runtime latency";
    }

    float result = ((float)(num_of_image) * 1000.0f / (float)latency);

    return result;
  }

  public float multi_instance_evaluate_performance() {
    int num_of_image = 0;
    long time = 0;
    long duration = m_duration * 1000;
    long start_time;

    for (time = 0, start_time = System.currentTimeMillis();
         time < duration; num_of_image += getBatchSize()) {
      runInference();
      time = System.currentTimeMillis() - start_time;
    }

    float result = ((float)num_of_image * 1000.0f / (float)time);

    return result;
  }

//
//  public void inference_frames(int frames) {
//    for (int iter = frames / getBatchSize(); iter > 0; --iter) {
//      runInference();
//    }
//  }

  /**
   * Get the name of the model file stored in Assets.
   *
   * @return
   */
  protected abstract String getModelPath();

  /**
   * Get the name of the label file stored in Assets.
   *
   * @return
   */
  protected abstract String getLabelPath();

  /**
   * Get the image size along the x axis.
   *
   * @return
   */
  public abstract int getImageSizeX();

  /**
   * Get the image size along the y axis.
   *
   * @return
   */
  public abstract int getImageSizeY();

  /**
   * Get the number of bytes that is used to store a single color channel value.
   *
   * @return
   */
  protected abstract int getNumBytesPerChannel();

  /**
   * Get the number of bytes that is used to store a single color channel value.
   *
   * @return
   */
  protected abstract int getBatchSize();

  /**
   * Add pixelValue to byteBuffer.
   *
   * @param pixelValue
   */
  protected abstract void addPixelValue(int pixelValue);

  /**
   * Read the probability value for the specified label This is either the original value as it was
   * read from the net's output or the updated value after the filter was applied.
   *
   * @param labelIndex
   * @return
   */
  protected abstract float getProbability(int labelIndex);

  /**
   * Set the probability value for the specified label.
   *
   * @param labelIndex
   * @param value
   */
  protected abstract void setProbability(int labelIndex, Number value);

  /**
   * Get the normalized probability value for the specified label. This is the final value as it
   * will be shown to the user.
   *
   * @return
   */
  protected abstract float getNormalizedProbability(int labelIndex);

  /**
   * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
   * provided by getProbability().
   *
   * <p>This additional method is necessary, because we don't have a common base for different
   * primitive data types.
   */
  protected abstract void runInference();

  public abstract void initalize_model(Activity activity) throws IOException;
  public abstract Inference clone();
  public abstract String name();

  /**
   * Get the total number of labels.
   *
   * @return
   */
  protected int getNumLabels() {
    if (labelList == null) {  // by LDY
      return 1;
    } else {
      return labelList.size();
    }
  }

  public String[] get_devices() {
    return tflite.get_devices();
  }

  protected Bitmap m_inputBitmap = null;

  public void input_bitmap(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    for (int batch = 0; batch < getBatchSize(); ++batch) {
      int pixel = 0;
      for (int i = 0; i < getImageSizeX(); ++i) {
        for (int j = 0; j < getImageSizeY(); ++j) {
          final int val = intValues[pixel++];
          addPixelValue(val);
        }
      }
    }

    m_inputBitmap = Bitmap.createBitmap(bitmap);
  }

  public Bitmap get_result_bitmap() {
    return m_inputBitmap;
  }

  private void qsort(int[] arr, float[] prob, int left, int right) {
    if (left < right) {
      int i, j, k;
      for (i = left + 1, j = left; i <= right; ++i) {
        if (prob[arr[i]] > prob[arr[left]]) {
          ++j;
          k = arr[j];
          arr[j] = arr[i];
          arr[i] = k;
        }
      }

      k = arr[j];
      arr[j] = arr[left];
      arr[left] = k;

      qsort(arr, prob, left, j - 1);
      qsort(arr, prob, j + 1, right);
    }
  }

  public String[] get_top_k_results(int k) {
    int[]   arr_idx   = new int[getNumLabels()];
    float[] arr_prob  = new float[getNumLabels()];
    String[] ret = new String[k];

    if (labelList == null) {
      return null;
    }

    for (int i = 0; i < getNumLabels(); ++i) {
      arr_idx[i] = i;
      arr_prob[i] = getProbability(i);
    }

    qsort(arr_idx, arr_prob, 0, getNumLabels() - 1);

    for (int i = 0; i < k; ++i) {
      ret[i] = String.format("%s (prob : %.3f)", labelList.get(arr_idx[i]), arr_prob[arr_idx[i]]);
    }

    return ret;
  }

/*
 * by LDY
 * the below source code is realted to image classification workload
 *
  void classifyFrame(Bitmap bitmap, SpannableStringBuilder builder) {
    if (tflite == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      builder.append(new SpannableString("Uninitialized Classifier."));
    }
    convertBitmapToByteBuffer(bitmap);
    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    runInference();
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // Smooth the results across frames.
    applyFilter();

    // Print the results.
    printTopKLabels(builder);
    long duration = endTime - startTime;
    SpannableString span = new SpannableString(duration + " ms\n");
    span.setSpan(new ForegroundColorSpan(android.graphics.Color.LTGRAY), 0, span.length(), 0);
    builder.append(span);
  }



  void applyFilter() {
    int numLabels = getNumLabels();

    // Low pass filter `labelProbArray` into the first stage of the filter.
    for (int j = 0; j < numLabels; ++j) {
      filterLabelProbArray[0][j] +=
          FILTER_FACTOR * (getProbability(j) - filterLabelProbArray[0][j]);
    }
    // Low pass filter each stage into the next.
    for (int i = 1; i < FILTER_STAGES; ++i) {
      for (int j = 0; j < numLabels; ++j) {
        filterLabelProbArray[i][j] +=
            FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
      }
    }

    // Copy the last stage filter output back to `labelProbArray`.
    for (int j = 0; j < numLabels; ++j) {
      setProbability(j, filterLabelProbArray[FILTER_STAGES - 1][j]);
    }
  }

  private void printTopKLabels(SpannableStringBuilder builder) {
    for (int i = 0; i < getNumLabels(); ++i) {
      sortedLabels.add(
              new AbstractMap.SimpleEntry<>(labelList.get(i), getNormalizedProbability(i)));
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }
    }

    final int size = sortedLabels.size();
    for (int i = 0; i < size; i++) {
      Map.Entry<String, Float> label = sortedLabels.poll();
      SpannableString span =
              new SpannableString(String.format("%s: %4.2f\n", label.getKey(), label.getValue()));
      int color;
      // Make it white when probability larger than threshold.
      if (label.getValue() > GOOD_PROB_THRESHOLD) {
        color = android.graphics.Color.WHITE;
      } else {
        color = SMALL_COLOR;
      }
      // Make first item bigger.
      if (i == size - 1) {
        float sizeScale = (i == size - 1) ? 1.25f : 0.8f;
        span.setSpan(new RelativeSizeSpan(sizeScale), 0, span.length(), 0);
      }
      span.setSpan(new ForegroundColorSpan(color), 0, span.length(), 0);
      builder.insert(0, span);
    }
  }
*/
}
