# from transformers import AutoModel, AutoConfig
# import torch
# import torch.nn as nn
# from transformers.models.vit_hybrid.modeling_vit_hybrid import *
#
# # Load the pre-trained model and configuration
# model_name = "bert-base-uncased"
# custom_config = AutoConfig.from_pretrained("google/vit-hybrid-base-bit-384")
#
# custom_config.num_channels = 512
# custom_config.backbone_featmap_shape = [1, 512, 20, 20]
# custom_config.image_size = 20
#
# hybrid_model = AutoModel.from_pretrained(model_name, config=custom_config)
#
#
# class ViTHybridModelInterface(nn.Module):
#     def __init__(self):
#         # before using this auto hybrid transformer model,
#         # go to models.auto.configuration_auto.py get item method to add the following
#         # if key == 'vit-hybrid':
#         #     key = 'vit_hybrid'
#         # and change every vit-hybrid to vit_hybrid
#         super().__init__()
#         self.model = hybrid_model
#         self.config = self.model.config
#     def forward(self, x):
#         output = self.model(x)
#         return output
