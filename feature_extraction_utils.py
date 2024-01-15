from PIL import Image
import torchvision.transforms.functional as TF
import yaml

# Custom layers import
from detr_custom_layers import *
from _yolo_custom_layers import *

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.cfg import get_cfg
from ultralytics.utils.ops import non_max_suppression
from ultralytics.nn.tasks import DetectionModel

class YOLOv8DetectionAndFeatureExtractorModel(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__(cfg, ch, nc, verbose)
        self.target_layer = 9  # Set the target layer number you want to inspect

    def print_layers(self):
        """
        Print all layers in the model.
        """
        print("Layers of the model:")
        for i, layer in enumerate(self.model.children()):
            print(f"Layer {i}: {layer}")

    def print_modules(self):
        """
        Print all modules in the model.
        """
        print("Modules in the model:")
        for name, module in self.model.named_modules():
            print(f"{name}: {module}")

    def custom_forward(self, x):
        """
        This method returns the output of the 9th layer (SPPF layer) exclusively.
        """
        for i, m in enumerate(self.model):
            x = m(x)  # run
            if i == 9:  # Check if it's the 9th layer (SPPF layer)
                return x.detach()  # Detach from the current graph and return the output

    # def custom_forward(self, x):
    #     """
    #     This is a modified version of the original _forward_once() method in BaseModel,
    #     found in ultralytics/nn/tasks.py.
    #     The original method returns only the detection output, while this method returns
    #     both the detection output and the features extracted by the last convolutional layer.
    #     """
    #     y = []
    #     features = None
    #     for m in self.model:
    #         if m.f != -1:  # if not from previous layer
    #             x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
    #         if torch.is_tensor(x):
    #             features = x  # keep the last tensor as features
    #         x = m(x)  # run
    #         if torch.is_tensor(x):
    #             features = x  # keep the last tensor as features
    #         y.append(x if m.i in self.save else None)  # save output
    #     if torch.is_tensor(x):
    #         features = x  # keep the last tensor as features
    #     return features, x  # return features and detection output


def create_yolov8_model(model_name_or_path, nc, class_names):
    ckpt = None
    if str(model_name_or_path).endswith('.pt'):
        weights, ckpt = attempt_load_one_weight(model_name_or_path)
        cfg = ckpt['model'].yaml
    else:
        cfg = model_name_or_path
    model = YOLOv8DetectionAndFeatureExtractorModel(cfg, nc=nc, verbose=True)
    if weights:
        model.load(weights)
    model.nc = nc
    model.names = class_names  # attach class names to model
    args = get_cfg(overrides={'model': model_name_or_path})
    model.args = args  # attach hyperparameters to model
    return model

def load_model_config(yaml_path):
    """
    Load number of classes (nc) and class names from a YAML file.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data.get('names', [])
    return len(class_names), class_names
def main():
    # Configuration
    model_path = 'runs/detect/train/weights/best.pt'  # Path to the model weight file
    image_path = 'datasets/mel_test1.jpg'  # Path to the image file
    yaml_path = 'datasets/combined_dataset/data.yaml'  # Path to the YAML file with class names

    # Load number of classes and class names from YAML
    nc, class_names = load_model_config(yaml_path)
    # Create the model
    model = create_yolov8_model(model_path, nc, class_names)

    # model.print_layers()
    # model.print_modules()

    # Load and process the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0)  # Add batch dimension

    # Usage
    # Assuming you have initialized the model and have an img_tensor to input
    layer9_output = model.custom_forward(img_tensor)

    # print("Layer 9 Output:", layer9_output)
    print("Shape of Layer 9 Output:", layer9_output.shape)

    # ###################################################################################################################
    # DETR MODEL VERSION
    # ###################################################################################################################
    embedding = EmbeddingLayer()
    encoder_decoder = DETREncoderDecoder() # outputting encoder_hidden_states of the same shape (you can consider these as image features)
    # flattened_features, flattened_mask, flattened_object_queries, batch_size = embedding(layer9_output)
    embedding_output = embedding(layer9_output)
    # print("Flat Features: ", embedding_output[0].shape)
    # print("Flat Mask: ", embedding_output[1].shape)
    # print("Flat Object queries: ", embedding_output[2].shape)
    # print("Batch Size: ", embedding_output[3])

    enc_outcome = encoder_decoder(embedding_output)
    print("Encoder Outcome: ", enc_outcome.shape)

    layer9_channels = layer9_output.shape[1]
    detr_channels = enc_outcome.shape[1]
    output_channels = 256

    # Initialize the gate layer
    gate_layer = GatedFusion(layer9_channels, detr_channels, output_channels)

    # Forward pass through the gate layer
    gated_output = gate_layer(layer9_output, enc_outcome)
    print("Gated Output Shape: ", gated_output.shape)

    # ###################################################################################################################
    # HYBRID TRANSFORMER MODEL VERSION
    # ###################################################################################################################
    # hybrid_transformer_model = ViTHybridModelInterface()
    # output = hybrid_transformer_model(layer9_output)
    # print(output)
    # print("Shape of Transformer Model Output:", output.shape)

    # ###################################################################################################################
    # YOLOv8 feature extraction
    # ###################################################################################################################
    # # Forward pass through the model
    # with torch.no_grad():
    #     features, detections = model.custom_forward(img_tensor)
    #
    # assert type(detections) == list or type(detections) == tuple
    # assert len(detections) == 3 or len(detections) == 2
    # if len(detections) == 2:
    #     print('YOLOv8 output in evaluation mode')
    #     yolov8_predictions = detections[0]
    #     yolov8_features = detections[1]
    #     print(f'yolov8_predictions.shape = {yolov8_predictions.shape}')
    #     yolov8_predictions = non_max_suppression(yolov8_predictions.detach(),
    #                                              conf_thres=0.1, iou_thres=0.1,
    #                                              max_det=1)
    #     print(f'len(yolov8_predictions) (after NMS) = {len(yolov8_predictions)}')
    # else:
    #     print('YOLOv8 output in training mode')
    #     yolov8_predictions = None
    #     yolov8_features = detections
    #
    # # Process detections
    # detections = non_max_suppression(detections)[0]  # Apply NMS
    #
    # # Do something with the features and detections
    # print("Features shape:", features.shape)
    # for i, det in enumerate(detections):
    #     print(f"i: {i}, Detection {det}")

if __name__ == '__main__':
    main()