package inference.resnet_v1_50;

import android.app.Activity;

import java.io.IOException;

import inference.Inference;

public class Resnet50FloatInference extends Inference {
    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
     * This isn't part of the super class, because we need a primitive array here.
     */
    private float[][] labelProbArray = null;

    @Override
    protected String getModelPath() {
        // you can download this file from
        // https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip
        return "resnet_v1_50_1.tflite"; // by LDY
    }

    @Override
    protected int getBatchSize() {
        return 1;
    }

    @Override
    public String name() {
        return String.format("Resnet50 Float Model (Batch %d)", getBatchSize());
    }

    @Override
    public Inference clone() {
        return new Resnet50FloatInference();
    }

    @Override
    protected String getLabelPath() {
        return "labels_imagenet_slim.txt";
    }

    @Override
    public int getImageSizeX() {
        return 224;
    }

    @Override
    public int getImageSizeY() {
        return 224;
    }

    @Override
    protected int getNumBytesPerChannel() {
        // a 32bit float value requires 4 bytes
        return 4;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.putFloat(((pixelValue >> 16) & 0xFF));
        imgData.putFloat(((pixelValue >> 8) & 0xFF));
        imgData.putFloat((pixelValue & 0xFF));
    }

    @Override
    protected float getProbability(int labelIndex) {
        return labelProbArray[0][labelIndex];
    }

    @Override
    protected void setProbability(int labelIndex, Number value) {
        labelProbArray[0][labelIndex] = value.floatValue();
    }

    @Override
    protected float getNormalizedProbability(int labelIndex) {
        // TODO the following value isn't in [0,1] yet, but may be greater. Why?
        return getProbability(labelIndex);
    }

    @Override
    protected void runInference() { tflite.run(imgData, labelProbArray);  }

    @Override
    public void initalize_model(Activity activity) throws IOException {
        load_model(activity);
        labelProbArray = new float[getBatchSize()][getNumLabels()];
    }
}
