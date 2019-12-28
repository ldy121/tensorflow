package inference;

import android.app.Activity;

import java.io.IOException;

public class DummyInference extends Inference {
    @Override
    protected String getModelPath() {
        return "dummy.tflite";
    }

    @Override
    protected String getLabelPath() {
        return null;
    }

    @Override
    public int getImageSizeX() {
        return 1;
    }

    @Override
    public int getImageSizeY() {        return 1;    }

    @Override
    protected int getNumBytesPerChannel() {
        return 1;
    }

    @Override
    protected int getBatchSize() {
        return 1;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
    }

    @Override
    protected float getProbability(int labelIndex) {
        return 0;
    }

    @Override
    protected void setProbability(int labelIndex, Number value) {
    }

    @Override
    protected float getNormalizedProbability(int labelIndex) {
        return 0;
    }

    @Override
    protected void runInference() {
    }

    @Override
    public void initalize_model(Activity activity) throws IOException {
        load_model(activity);
    }

    @Override
    public Inference clone() {
        return new DummyInference();
    }

    @Override
    public String name() {
        return null;
    }
}
