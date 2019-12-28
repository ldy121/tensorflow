package network;

public class ImageRegionNetwork extends Network {
    public ImageRegionNetwork() {
        add_model("SSD-Mobilenet V1", "SSDMobilenetV1");
        add_model("DeepLab V3+ (Resnet50)", "DeepLabV3PlusResnet50");
    }
}
