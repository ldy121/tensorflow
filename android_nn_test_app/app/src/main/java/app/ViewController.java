package app;

import android.app.Activity;

import inference.Inference;

public abstract class ViewController {
    protected Activity m_acitivity;

    public ViewController(Activity activity) {
        m_acitivity = activity;
    }

    public abstract void onActive();
    public abstract int get_content_resource_id();
    public abstract String get_controller_name();
    public abstract ObserverDone onRun(int id, Inference inference[], String log_header);

    public void run_workload(int id, Inference inference[], int iteration, String log_header) {
        ObserverDone done = onRun(id, inference, log_header);
        WorkloadDocument document = new WorkloadDocument(inference, iteration);
        document.start_workload();
        done.update(document);
    }
}
