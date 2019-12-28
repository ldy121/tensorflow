package inference.ssd_mobilenet_v1;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import inference.Inference;

public class SSDMobilenetQuantizedV1 extends Inference {
    private final static int num_of_post_process = 4;
    private final static int num_of_bounding_box = 10;
    private Map<Integer, Object> m_output = new HashMap<>();

    private final static int m_location = 0;
    private final static int m_classes = 1;
    private final static int m_scores = 2;
    private final static int m_numDetections = 3;

    @Override
    protected String getModelPath() {
        return "ssd_mobilenet_v1_quant_1.tflite"; // by LDY
    }

    @Override
    protected int getBatchSize() {
        return 1;
    }

    @Override
    public String name() {
        return String.format("SSD MobilenetV1 Quant Model (Batch %d)", getBatchSize());
    }

    @Override
    protected String getLabelPath() {   return "ssd_mobilenet_v1_mscoco_label.txt"; }

    @Override
    public int getImageSizeX() {
        return 300;
    }

    @Override
    public int getImageSizeY() {
        return 300;
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
        return 0;
    }

    @Override
    protected void setProbability(int labelIndex, Number value) {
    }

    @Override
    protected float getNormalizedProbability(int labelIndex) {
        return getProbability(labelIndex);
    }

    @Override
    protected void runInference() {
        Object[] inputArray = {imgData};

        tflite.runForMultipleInputsOutputs(inputArray, m_output);
    }


    @Override
    public void initalize_model(Activity activity) throws IOException {
        load_model(activity);

        float[][][] outputLocations = new float[getBatchSize()][num_of_bounding_box][num_of_post_process];
        float[][] outputClasses = new float[getBatchSize()][num_of_bounding_box];
        float[][] outputScores = new float[getBatchSize()][num_of_bounding_box];
        float[] numDetections = new float[getBatchSize()];

        m_output.put(m_location, outputLocations);
        m_output.put(m_classes, outputClasses);
        m_output.put(m_scores, outputScores);
        m_output.put(m_numDetections, numDetections);

        tflite.setNumThreads(num_of_post_process);
    }

    @Override
    public Inference clone() {
        return new SSDMobilenetQuantizedV1();
    }

    @Override
    public Bitmap get_result_bitmap() {
        if (m_inputBitmap == null) {
            return null;
        }

        Canvas canvas = new Canvas(m_inputBitmap);
        Paint paint = new Paint();
        Paint text_paint = new Paint();

        int color = Color.BLACK;
        int labelOffset = 1;
        int boundary_width = 5;
        float threshold = 0.5f;

        paint.setStrokeWidth(boundary_width);
        paint.setStyle(Paint.Style.STROKE);

        for (int batch = 0; batch < getBatchSize(); ++batch) {
            float[][] outputLocations = ((float[][][]) m_output.get(m_location))[batch];
            float[] outputClasses = (((float[][])(m_output.get(m_classes)))[batch]);
            float[] outputScores = (((float[][])(m_output.get(m_scores)))[batch]);

            for (int i = 0; i < num_of_bounding_box; ++i) {
                float score = outputScores[i];
                if (score > threshold) {
                    int margin = boundary_width * 4;

                    RectF detection = new RectF(
                            outputLocations[i][1] * getImageSizeX(),
                            outputLocations[i][0] * getImageSizeY(),
                            outputLocations[i][3] * getImageSizeX(),
                            outputLocations[i][2] * getImageSizeY()
                    );

                    paint.setColor(color);
                    text_paint.setColor(color);

                    String label = String.format("%s (probability : %f)",
                            labelList.get((int) outputClasses[i] + labelOffset), score);

                    canvas.drawRect(detection, paint);
                    canvas.drawText(label, detection.left + margin, detection.top + margin, text_paint);

                    color += 128;
                }
            }
        }

        return m_inputBitmap;
    }
}
