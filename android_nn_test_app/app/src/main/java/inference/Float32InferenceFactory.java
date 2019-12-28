package inference;

import android.app.Activity;

import inference.deeplab_v3_plus_resnet_v1_50.DeeplabV3PlusResNet50FloatInference;
import inference.inception_v3.Inceptionv3FloatBatchInference;
import inference.inception_v3.Inceptionv3FloatInference;
import inference.mobilenet_v1.MobilenetV1FloatBatchInference;
import inference.mobilenet_v1.MobilenetV1FloatInference;
import inference.resnet_v1_50.Resnet50FloatBatchInference;
import inference.resnet_v1_50.Resnet50FloatInference;
import inference.ssd_mobilenet_v1.SSDMobilenetFloatV1;

public class Float32InferenceFactory extends InferenceFactory {
    private static Float32InferenceFactory instance = new Float32InferenceFactory();

    private Float32InferenceFactory() {
        m_prototype.put("Inceptionv3", new Inceptionv3FloatInference());
        m_prototype.put("Inceptionv3_batch", new Inceptionv3FloatBatchInference());

        m_prototype.put("Resnet50", new Resnet50FloatInference());
        m_prototype.put("Resnet50_batch", new Resnet50FloatBatchInference());

        m_prototype.put("MobilenetV1", new MobilenetV1FloatInference());
        m_prototype.put("MobilenetV1_batch", new MobilenetV1FloatBatchInference());

        m_prototype.put("SSDMobilenetV1", new SSDMobilenetFloatV1());
        m_prototype.put("DeepLabV3PlusResnet50", new DeeplabV3PlusResNet50FloatInference());
        m_prototype.put("dummy", new DummyInference());
    }

    public static InferenceFactory get_instance() {
        return instance;
    }

    @Override
    public Inference createInference(Activity activity, String str) {
        Inference ret = createPrototypeInferece(activity, str);
        if (ret != null) {
            ret.useNNAPI();
        }
        return ret;
    }

    @Override
    public String get_name() {
        return new String("FP32");
    }
}
