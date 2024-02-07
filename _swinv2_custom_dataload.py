from transformers.utils import send_example_telemetry

model_checkpoint = "microsoft/swin-tiny-patch4-window7-224" # pre-trained model from which to fine-tune
batch_size = 32 # batch size for training and evaluation

send_example_telemetry("image_classification_notebook", framework="pytorch")
