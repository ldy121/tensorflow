package inference;

import android.app.Activity;

import inference.deeplab_v3_plus_resnet_v1_50.DeeplabV3PlusResNet50QuantizedInference;
import inference.inception_v3.Inceptionv3QuantizedBatchInference;
import inference.inception_v3.Inceptionv3QuantizedInference;
import inference.mobilenet_v1.MobilenetV1QuantizedBatchInference;
import inference.mobilenet_v1.MobilenetV1QuantizedInference;
import inference.resnet_v1_50.Resnet50QuantizedBatchInference;
import inference.resnet_v1_50.Resnet50QuantizedInference;
import inference.ssd_mobilenet_v1.SSDMobilenetQuantizedV1;

public class Int8InferenceFactory extends InferenceFactory {
    private static Int8InferenceFactory instance = new Int8InferenceFactory();

    private Int8InferenceFactory() {
        m_prototype.put("Inceptionv3", new Inceptionv3QuantizedInference());
        m_prototype.put("Inceptionv3_batch", new Inceptionv3QuantizedBatchInference());

        m_prototype.put("Resnet50", new Resnet50QuantizedInference());
        m_prototype.put("Resnet50_batch", new Resnet50QuantizedBatchInference());

        m_prototype.put("MobilenetV1", new MobilenetV1QuantizedInference());
        m_prototype.put("MobilenetV1_batch", new MobilenetV1QuantizedBatchInference());

        m_prototype.put("SSDMobilenetV1", new SSDMobilenetQuantizedV1());
        m_prototype.put("DeepLabV3PlusResnet50", new DeeplabV3PlusResNet50QuantizedInference());
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
        return new String("Int8");
    }
}
