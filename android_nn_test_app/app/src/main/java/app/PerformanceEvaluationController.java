package app;

import android.app.Activity;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.HashMap;

import group.ap.android_nn_benchmark.R;
import inference.Inference;

public class PerformanceEvaluationController extends ViewController {
    private static int device_id_start = 123456789;

    private HashMap<Integer, String> m_mapDevice = new HashMap<Integer, String>();
    private HashMap<Integer, Integer>   m_zero_ratio = new HashMap<Integer, Integer>();
    HashMap<Integer, String> m_mapExecutionMode = new HashMap<Integer, String>();

    private String m_device;
    private String[] m_devices;

    private int m_input_zero_ratio;
    private String m_exePreference;
    private String m_result = new String();

    private void init_button_zero_ratio() {
        m_zero_ratio.put(R.id.run_random_workload, 0);
        m_zero_ratio.put(R.id.run_zero_workload, 100);

        m_input_zero_ratio = 0;
    }

    public PerformanceEvaluationController(Activity activity, Inference dummy) {
        super(activity);
        m_devices = dummy.get_devices();
        init_button_zero_ratio();
    }

    private void init_device_table() {

        RadioGroup device_radio = m_acitivity.findViewById(R.id.device_type);
        RadioButton defaultButton = new RadioButton(m_acitivity);

        device_radio.removeAllViews();

        defaultButton.setText("default");
        defaultButton.setTextSize(10);
        defaultButton.setId(++device_id_start);
        device_radio.addView(defaultButton);
        m_mapDevice.put(device_id_start, null);

        if (m_devices != null) {
            for (int i = 0; i < m_devices.length; ++i) {
                if (!m_devices[i].isEmpty()) {

                    int id = (++device_id_start);
                    RadioButton radioButton = new RadioButton(m_acitivity);

                    radioButton.setText(m_devices[i]);
                    radioButton.setTextSize(10);
                    radioButton.setId(id);

                    device_radio.addView(radioButton);
                    m_mapDevice.put(id, m_devices[i].split((":"))[0]);
                }
            }
        }

        device_radio.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                m_device = m_mapDevice.get(checkedId);
            }
        });

        defaultButton.setChecked(true);
    }

    private void init_exe_table() {
        m_mapExecutionMode.put(R.id.radio_exe_default, null);
        m_mapExecutionMode.put(R.id.radio_exe_low_power, "low_power");
        m_mapExecutionMode.put(R.id.radio_exe_sustained_speed, "sustained_speed");
        m_mapExecutionMode.put(R.id.radio_exe_fast_single_answer, "fast_single_answer");

        RadioGroup exe_radio_group = m_acitivity.findViewById(R.id.exec_preference);
        exe_radio_group.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                m_exePreference = m_mapExecutionMode.get(checkedId);
            }
        });

        RadioButton default_button = m_acitivity.findViewById(R.id.radio_exe_fast_single_answer);
        default_button.setChecked(true);
    }


    @Override
    public void onActive() {
        Button random_button = m_acitivity.findViewById(R.id.run_random_workload);
        Button zero_button = m_acitivity.findViewById(R.id.run_zero_workload);
        TextView result = m_acitivity.findViewById(R.id.performance_result);

        random_button.setOnClickListener((View.OnClickListener) m_acitivity);
        zero_button.setOnClickListener((View.OnClickListener) m_acitivity);
        result.setMovementMethod(new ScrollingMovementMethod());
        result.setText(m_result);

        init_device_table();
        init_exe_table();
    }

    @Override
    public int get_content_resource_id() {
        return R.layout.performance_evaluation;
    }

    @Override
    public String get_controller_name() {
        return "Performance Evaluation";
    }

    class PerformanceEvaluationControllerObserver extends ObserverDone {
        private String m_log_header;
        public PerformanceEvaluationControllerObserver(Activity activity, String log_header) {
            super(activity);
            m_log_header = log_header;
        }

        @Override
        void update(final WorkloadDocument document) {
            final String test_info = String.format("%s\nDevice : %s / exe preference : %s / zero ratio : %d%%\n",
                    m_log_header, (m_device != null) ? m_device : "default", m_exePreference, m_input_zero_ratio);

            m_acitivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    TextView output = m_acitivity.findViewById(R.id.performance_result);
                    String result = new String(test_info);
                    ArrayList<Float> results = document.get_perf_result();
                    String perf_metric = document.get_perf_metric();
                    for (int i = 0; i < results.size(); ++i) {
                        float fps = ((float) results.get(i));
                        result += String.format("#%d test result - %f (Perf metric : %s)\n", i + 1, fps, perf_metric);
                    }
                    output.setText(result + "\n" + output.getText());

                    m_result = output.getText().toString();
                }
            });
        }
    }

    @Override
    public ObserverDone onRun(int id, Inference inferences[], String log_header) {
        m_input_zero_ratio = m_zero_ratio.get(id);

        for (int i = 0; i < inferences.length; ++i) {
            inferences[i].generate_input(m_input_zero_ratio);
            inferences[i].setExecutionPreference(m_exePreference);
            inferences[i].setAccelerator(m_device);
        }

        return (new PerformanceEvaluationControllerObserver(m_acitivity, log_header));
    }
}
