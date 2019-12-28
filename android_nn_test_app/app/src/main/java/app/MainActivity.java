package app;

import android.content.Context;
import android.os.Bundle;

import android.os.PowerManager;
import android.view.LayoutInflater;
import android.view.View;

import androidx.core.view.GravityCompat;
import androidx.appcompat.app.ActionBarDrawerToggle;

import android.view.MenuItem;

import com.google.android.material.navigation.NavigationView;

import androidx.drawerlayout.widget.DrawerLayout;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.view.Menu;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.HashMap;

import group.ap.android_nn_benchmark.R;
import inference.Float16InferenceFactory;
import inference.Float32InferenceFactory;
import inference.Inference;
import inference.InferenceFactory;
import inference.Int8InferenceFactory;
import network.ClassificationNetwork;
import network.ImageRegionNetwork;
import network.ObjectDetectionNetwork;
import network.SegmentationNetwork;

public class MainActivity extends AppCompatActivity
        implements NavigationView.OnNavigationItemSelectedListener, View.OnClickListener {
    HashMap<Integer, WorkloadFacsade> m_workloadTable = new HashMap<Integer, WorkloadFacsade>();
    HashMap<Integer, InferenceFactory>  m_factoryTable = new HashMap<Integer, InferenceFactory>();

    private static final String TAG = "ANDROID_NN_TEST";
    private static int resource_id = 987654321;

    String m_modelName = new String("Inceptionv3");
    InferenceFactory m_factory;
    WorkloadFacsade m_workload;

    Boolean m_exe_hw_time = false;
    PowerManager.WakeLock m_wakelock;

    private void init_workload_table() {
        Inference dummy = m_factory.createInference(this, "dummy");
        WorkloadFacsade workload;

        m_workload =    // to configure initial workload
        workload = new WorkloadFacsade(new ClassificationNetwork(), new PerformanceEvaluationController(this, dummy));
        m_workloadTable.put(R.id.performance_classification, workload);

        workload = new WorkloadFacsade(new ObjectDetectionNetwork(), new PerformanceEvaluationController(this, dummy));
        m_workloadTable.put(R.id.performance_object_detection, workload);

        workload = new WorkloadFacsade(new SegmentationNetwork(), new PerformanceEvaluationController(this, dummy));
        m_workloadTable.put(R.id.performance_segmentation, workload);

        workload = new WorkloadFacsade(new ImageRegionNetwork(), new ImageRegionNetworkController(this));
        m_workloadTable.put(R.id.validation_region, workload);

        workload = new WorkloadFacsade(new ClassificationNetwork(), new ClassificationNetworkController(this));
        m_workloadTable.put(R.id.validation_classifcation, workload);
    }

    private void init_factory_table() {
        m_factoryTable.put(R.id.radio_int8_inference, Int8InferenceFactory.get_instance());
        m_factoryTable.put(R.id.radio_fp16_inference, Float16InferenceFactory.get_instance());
        m_factoryTable.put(R.id.radio_fp32_inference, Float32InferenceFactory.get_instance());

        RadioGroup infer_group = findViewById(R.id.infer_type);
        infer_group.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                m_factory = m_factoryTable.get(checkedId);
            }
        });

        RadioButton radio = findViewById(R.id.radio_int8_inference);
        radio.setChecked(true);
    }


    private void load_networks() {
        RadioGroup network_radio = findViewById(R.id.network_list);
        ArrayList<String> networks = m_workload.get_network_resource();
        RadioButton first_button = null;

        network_radio.removeAllViews();

        for (int i = 0; i < networks.size(); ++i) {
            int id = resource_id++;
            RadioButton radioButton = new RadioButton((this));
            radioButton.setText(networks.get(i));
            radioButton.setTextSize(10);
            radioButton.setId(id);

            m_workload.register_network_resource_id(networks.get(i), id);

            network_radio.addView(radioButton);

            if (i == 0) {
                first_button = radioButton;
            }
        }

        network_radio.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                m_modelName = m_workload.get_network_model(checkedId);
            }
        });

        first_button.setChecked(true);
    }

    private void change_view() {
        LayoutInflater inflater = (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        FrameLayout frame = findViewById(R.id.dynamic_area);
        frame.removeAllViews();
        View view = inflater.inflate(m_workload.get_content_view_resource(), frame, false);
        frame.addView(view);
    }

    private void change_workload() {
        load_networks();
        change_view();
        m_workload.active_workload();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        DrawerLayout drawer = findViewById(R.id.drawer_layout);
        NavigationView navigationView = findViewById(R.id.nav_view);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(
                this, drawer, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawer.addDrawerListener(toggle);
        toggle.syncState();
        navigationView.setNavigationItemSelectedListener(this);

        CheckBox hw_checkbox = findViewById(R.id.enable_hw_time);
        hw_checkbox.setOnCheckedChangeListener(new CheckBox.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                m_exe_hw_time = isChecked;
            }
        });
        hw_checkbox.setChecked(true);

        init_factory_table();
        init_workload_table();

        change_workload();

        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        m_wakelock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, TAG);
    }

    @Override
    public void onBackPressed() {
        DrawerLayout drawer = findViewById(R.id.drawer_layout);
        if (drawer.isDrawerOpen(GravityCompat.START)) {
            drawer.closeDrawer(GravityCompat.START);
        } else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @SuppressWarnings("StatementWithEmptyBody")
    @Override
    public boolean onNavigationItemSelected(MenuItem item) {
        // Handle navigation view item clicks here.
        int id = item.getItemId();

        if (runClassifier) {
            Toast.makeText(this, "A test workload is in progress", Toast.LENGTH_SHORT).show();
            return true;
        }

        m_workload = m_workloadTable.get(id);
        change_workload();

        TextView textView = findViewById(R.id.nn_test_name);
        textView.setText(m_workload.get_workload_name());

        DrawerLayout drawer = findViewById(R.id.drawer_layout);
        drawer.closeDrawer(GravityCompat.START);
        return true;
    }

    private boolean runClassifier = false;

    @Override
    public void onClick(View v) {
        if (runClassifier == false) {
            runClassifier = true;
            m_wakelock.acquire();

            int thread = Integer.parseInt(((EditText)findViewById(R.id.text_workload)).getText().toString());
            int duration = Integer.parseInt(((EditText)findViewById(R.id.text_duration)).getText().toString());
            final int iteration = Integer.parseInt(((EditText)findViewById(R.id.text_iteration)).getText().toString());

            final Inference[] inferences = new Inference[thread];
            for (int i = 0; i < inferences.length; ++i) {
                inferences[i] = m_factory.createInference(this, m_modelName);
                inferences[i].set_duration(duration);
                inferences[i].set_hw_exe_time(m_exe_hw_time);
            }

            final int id = v.getId();
            final String log_header = String.format("Network name : %s(%s) / # of instance : %d / Duration : %d(sec)",
                    m_modelName, m_factory.get_name(), thread, duration);

            new Thread(new Runnable() {
                @Override
                public void run() {
                    m_workload.run_workload(id, inferences, iteration, log_header);

                    for (int i = 0; i < inferences.length; ++i) {
                        inferences[i].clone();
                    }

                    runClassifier = false;
                    m_wakelock.release();
                }
            }).start();
        } else {
            Toast.makeText(this, "A test workload is in progress", Toast.LENGTH_SHORT).show();
        }
    }
}
