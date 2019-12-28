package app;

import java.util.ArrayList;

import inference.Inference;
import network.Network;

class WorkloadFacsade {
    private Network m_network;
    private ViewController m_viewController;

    WorkloadFacsade(Network network, ViewController controller) {
        m_network = network;
        m_viewController = controller;
    }

    ArrayList<String> get_network_resource() {
        return m_network.get_resource_name();
    }

    void register_network_resource_id(String name, int id) {
        m_network.register_resource_id(name, id);
    }

    String get_network_model(int id) {
        return m_network.get_model(id);
    }

    int get_content_view_resource() {
        return m_viewController.get_content_resource_id();
    }

    String get_workload_name() {
        return m_viewController.get_controller_name();
    }

    void active_workload() { m_viewController.onActive(); }

    void run_workload(int id, Inference inference[], int iteration, String log_header) {
        m_viewController.run_workload(id, inference, iteration, log_header);
    }
}
