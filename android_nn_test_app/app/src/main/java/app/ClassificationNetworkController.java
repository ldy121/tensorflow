package app;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.w3c.dom.Text;

import group.ap.android_nn_benchmark.R;
import inference.Inference;

public class ClassificationNetworkController extends ViewController {
    public ClassificationNetworkController(Activity activity) {
        super(activity);
    }

    @Override
    public void onActive() {
        Button button = m_acitivity.findViewById(R.id.run_validation_classification);
        TextView result = m_acitivity.findViewById(R.id.classification_validataion_result);

        button.setOnClickListener((View.OnClickListener) m_acitivity);
        result.setMovementMethod(new ScrollingMovementMethod());
    }

    class ClassificationObserver extends ObserverDone {
        public ClassificationObserver(Activity activity) {
            super(activity);
        }

        @Override
        void update(final WorkloadDocument document) {
            m_acitivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    TextView resultView = m_acitivity.findViewById(R.id.classification_validataion_result);
                    String[] results = document.get_top_labels();
                    String output = new String();

                    for (int i = 0; i < results.length; ++i) {
                        output += (results[i] + "\n");
                    }

                    resultView.setText(output);
                }
            });
        }
    }

    @Override
    public ObserverDone onRun(int id, Inference[] inferences, String log_header) {
        final Inference infer = inferences[0];
        ImageView imageView = m_acitivity.findViewById(R.id.calssification_input_image);

        final BitmapDrawable bitmapDrawable = (BitmapDrawable) imageView.getDrawable();
        Bitmap bitmap = bitmapDrawable.getBitmap();

        Bitmap inputBitmap = Bitmap.createScaledBitmap(bitmap, infer.getImageSizeX(), infer.getImageSizeY(), true);
        infer.input_bitmap(inputBitmap);

        return new ClassificationObserver(m_acitivity);
    }

    @Override
    public int get_content_resource_id() {        return R.layout.classify_validation;    }

    @Override
    public String get_controller_name() {        return "Classification network validation";    }
}
