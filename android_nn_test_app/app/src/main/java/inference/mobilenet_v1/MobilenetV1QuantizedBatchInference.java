package inference.mobilenet_v1;

import android.app.Activity;

import java.io.IOException;

import inference.Inference;

public class MobilenetV1QuantizedBatchInference extends Inference {
    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
     * This isn't part of the super class, because we need a primitive array here.
     */
    private byte[][] labelProbArray = null;

    @Override
    protected String getModelPath() {
        return "mobilenet_v1_quant_16.tflite"; // by LDY
    }

    @Override
    protected int getBatchSize() {
        return 16;
    }

    @Override
    public String name() {
        return String.format("MobilenetV1 Quant Model (Batch %d)", getBatchSize());
    }

    @Override
    public Inference clone() {
        return new MobilenetV1QuantizedBatchInference();
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
        // the quantized model uses a single byte only
        return 1;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
        imgData.put((byte) (pixelValue & 0xFF));
    }

    @Override
    protected float getProbability(int labelIndex) {
        return labelProbArray[0][labelIndex];
    }

    @Override
    protected void setProbability(int labelIndex, Number value) {
        labelProbArray[0][labelIndex] = value.byteValue();
    }

    @Override
    protected float getNormalizedProbability(int labelIndex) {
        //  Makalu the Inception Quantization version is working on GPU
        //  Accuracy of network result will be very bad
        return (labelProbArray[0][labelIndex] & 0xff) / 255.0f;
    }

    @Override
    protected void runInference() {
        tflite.run(imgData, labelProbArray);
    }

    @Override
    public void initalize_model(Activity activity) throws IOException {
        load_model(activity);
        labelProbArray = new byte[getBatchSize()][getNumLabels()];
    }
}
