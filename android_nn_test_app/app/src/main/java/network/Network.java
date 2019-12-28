package network;

import java.util.ArrayList;
import java.util.HashMap;

public class Network {
    private HashMap<String, String>  m_resource_text_to_model_name  = new HashMap<String, String>();
    private HashMap<Integer, String> m_resource_id_to_model_name    = new HashMap<Integer, String>();
    private ArrayList<String>        m_resource_text                = new ArrayList<String>();

    protected void add_model(String resource_text, String network_name) {
        m_resource_text.add(resource_text);
        m_resource_text_to_model_name.put(resource_text, network_name);
    }

    public void register_resource_id(String resource_text, int id) {
        String network_model = m_resource_text_to_model_name.get(resource_text);
        m_resource_id_to_model_name.put(id, network_model);
    }

    public String get_model(int id) {
        return m_resource_id_to_model_name.get(id);
    }

    public ArrayList<String> get_resource_name() {
        return m_resource_text;
    }
}
