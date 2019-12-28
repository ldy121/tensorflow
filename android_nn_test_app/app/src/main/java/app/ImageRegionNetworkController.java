package app;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import group.ap.android_nn_benchmark.R;
import inference.Inference;

public class ImageRegionNetworkController extends ViewController {
    private Bitmap m_inputBitmap;

    @Override
    public void onActive() {
        Button button = m_acitivity.findViewById(R.id.run_validation_region);
        button.setOnClickListener((View.OnClickListener) m_acitivity);
    }

    @Override
    public ObserverDone onRun(int id, Inference[] inferences, String log_header) {
        Inference infer = inferences[0];
        ImageView imageView = m_acitivity.findViewById(R.id.input_image_region);

        BitmapDrawable bitmapDrawable = (BitmapDrawable) imageView.getDrawable();
        m_inputBitmap = bitmapDrawable.getBitmap();

        Bitmap inputBitmap = Bitmap.createScaledBitmap(m_inputBitmap, infer.getImageSizeX(), infer.getImageSizeY(), true);
        infer.input_bitmap(inputBitmap);

        return new ImageRegionObserver(m_acitivity);
    }

    class ImageRegionObserver extends ObserverDone {
        public ImageRegionObserver(Activity activity) {
            super(activity);
        }

        @Override
        void update(final WorkloadDocument document) {
            m_acitivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    ImageView resultView = m_acitivity.findViewById(R.id.result_image_region);

                    Bitmap output = document.get_result_bitmap();
                    Bitmap outputBitmap = Bitmap.createScaledBitmap(output, m_inputBitmap.getWidth(), m_inputBitmap.getHeight(), true);

                    resultView.setImageBitmap(outputBitmap);
                }
            });
        }
    }

    public ImageRegionNetworkController(Activity activity) {    super(activity);    }
    @Override
    public int get_content_resource_id() { return R.layout.image_region_validation; }
    @Override
    public String get_controller_name() {   return "ImageRegionNetwork validation"; }
}
