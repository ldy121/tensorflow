package inference;

import android.app.Activity;

import java.io.IOException;
import java.util.HashMap;

public abstract class InferenceFactory {
    public abstract Inference createInference(Activity activity, String str);
    public abstract String get_name();

    protected HashMap<String, Inference> m_prototype = new HashMap<String, Inference>();
    protected InferenceFactory m_instance;

    protected Inference createPrototypeInferece(Activity activity, String str)
    {
        Inference inference = m_prototype.get(str);
        if (inference != null) {
            Inference ret = inference.clone();
            try {
                ret.initalize_model(activity);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return ret;
        } else {
            return null;
        }
    }
}
