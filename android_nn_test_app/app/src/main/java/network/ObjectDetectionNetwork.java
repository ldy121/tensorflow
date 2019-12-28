package network;

public class ObjectDetectionNetwork extends Network {
    public ObjectDetectionNetwork() {
        add_model("SSD-Mobilenet V1", "SSDMobilenetV1");
    }
}
