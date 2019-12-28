package network;

public class SegmentationNetwork extends Network {
    public SegmentationNetwork() {
        add_model("DeepLab V3+ (Resnet50)", "DeepLabV3PlusResnet50");
    }
}
