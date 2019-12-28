package app;

import android.app.Activity;

public abstract class ObserverDone {
    protected Activity m_activity;
    abstract void update(WorkloadDocument document);
    public ObserverDone(Activity activity) {
        m_activity = activity;
    }
}
