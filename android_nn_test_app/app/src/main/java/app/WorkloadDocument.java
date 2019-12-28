package app;

import android.graphics.Bitmap;
import android.os.Handler;
import java.util.ArrayList;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import inference.Inference;

public class WorkloadDocument {
    private int m_iteration;

    // From example source code
    private final Object lock = new Object();
    private ArrayList<Float> m_test_result;

    private float m_inference_result = 0.0f;
    private Bitmap m_bitmap;

    private Inference[] m_inference;
    public WorkloadDocument(Inference[] inference, int iteration) {
        m_iteration = iteration;
        m_inference = inference;
    }

    class run_inference implements Runnable {
        private Inference m_inference;

        public run_inference (Inference inference) {
            m_inference = inference;
        }

        @Override
        public void run() {
            for (int i = 0; i < m_iteration; ++i) {
                float result = 0.0f;

                try {
                    m_start_barrier.await();
                } catch (BrokenBarrierException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                result = m_inference.multi_instance_evaluate_performance();
                synchronized (lock) {
                    m_inference_result += result;
                }

                try {
                    m_end_barrier.await();
                } catch (BrokenBarrierException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    };

    private CyclicBarrier m_start_barrier;
    private CyclicBarrier m_end_barrier;

    /** Starts a background thread and its {@link Handler}. */
    private String multi_instance_test() {
        Thread backgroundThreads[] = new Thread[m_inference.length];

        m_start_barrier = new CyclicBarrier(m_inference.length + 1);
        m_end_barrier = new CyclicBarrier(m_inference.length + 1);

        for (int i = 0; i < m_inference.length; ++i) {
            run_inference run_infer = new run_inference(m_inference[i]);

            backgroundThreads[i] = new Thread(run_infer);
            backgroundThreads[i].start();
        }

        for (int ii = 0; ii < m_iteration; ++ii) {
            synchronized (lock) {
                m_inference_result = 0.0f;
            }
            try {
                m_start_barrier.await();
            } catch (BrokenBarrierException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            try {
                m_end_barrier.await();
            } catch (BrokenBarrierException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            synchronized (lock) {
                m_test_result.add(m_inference_result);
            }
        }

        for (int i = 0; i < m_inference.length; ++i) {
            try {
                backgroundThreads[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return ("Throughput App runtime");
    }

    private String single_instance_test() {
        Inference infer = m_inference[0];
        for (int i = 0; i < m_iteration; ++i) {
            m_inference_result = infer.single_instance_evaluate_performance();
            m_test_result.add(m_inference_result);
        }

        return infer.get_perf_metric();
    }

    public void start_workload() {
        String perf_metric;

        m_test_result = new ArrayList<Float>();

        if (m_inference.length == 1) {
            perf_metric = single_instance_test();
        } else {
            perf_metric = multi_instance_test();
        }

        m_perf_metric = perf_metric;
        m_bitmap = m_inference[0].get_result_bitmap();
        m_labels = m_inference[0].get_top_k_results(3);
    }

    private String m_perf_metric;
    private String m_labels[];
    public String[] get_top_labels()            {   return m_labels;        }
    public String get_perf_metric()             {   return m_perf_metric;   }
    public ArrayList<Float> get_perf_result()   {   return m_test_result;   }
    public Bitmap get_result_bitmap()           {   return m_bitmap;        }
}