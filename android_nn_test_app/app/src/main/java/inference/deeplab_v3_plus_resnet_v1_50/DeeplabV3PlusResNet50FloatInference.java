package inference.deeplab_v3_plus_resnet_v1_50;

import android.app.Activity;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.IOException;

import inference.Inference;

public class DeeplabV3PlusResNet50FloatInference extends Inference {
    private long[][][] segmentation = null;
    private final static int width   = 513;
    private final static int height  = 513;

    @Override
    protected String getModelPath() {
        return "deeplabv3_plus_resnet50_1.tflite"; // by LDY
    }

    @Override
    protected int getBatchSize() {
        return 1;
    }

    @Override
    public String name() {
        return String.format("DeepLabV3+ ResNet50 Float Model (Batch %d)", getBatchSize());
    }

    @Override
    protected String getLabelPath() {
        return null;
    }

    @Override
    public int getImageSizeX() {
        return 513;
    }

    @Override
    public int getImageSizeY() {
        return 513;
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
        return 0;
    }

    @Override
    protected void setProbability(int labelIndex, Number value) {}

    @Override
    protected float getNormalizedProbability(int labelIndex) {
        return getProbability(labelIndex);
    }

    @Override
    protected void runInference() { tflite.run(imgData, segmentation);  }

    @Override
    public void initalize_model(Activity activity) throws IOException {
        load_model(activity);
        segmentation = new long[getBatchSize()][height][width];
    }

    @Override
    public Inference clone() {
        return new DeeplabV3PlusResNet50FloatInference();
    }

    @Override
    public Bitmap get_result_bitmap() {
        for (int batch = 0; batch < getBatchSize(); ++batch) {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    Log.d("LDY", "" + segmentation[batch][i][j]);
                }
            }
        }

        return m_inputBitmap;
    }
}
