package network;

public class ClassificationNetwork extends Network {
    public ClassificationNetwork() {
        add_model("Mobilenet V1", "MobilenetV1");
        add_model("Mobilenet V1 (batch)", "MobilenetV1_batch");
        add_model("Inception V3", "Inceptionv3");
        add_model("Inception V3 (batch)", "Inceptionv3_batch");
        add_model("Resnetv1 50", "Resnet50");
        add_model("Resnetv1 50 (batch)", "Resnet50_batch");
    }
}
