import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from transformers.models.detr.modeling_detr import *

### NOTE ###########################################################################################################
# 1. after implementation, you must import and include them in all at the file "ultralytics/nn/modules/__init__.py"
# from custom_layers import FakeConv, ModifiedFakeBlock
# __all__ = (..., 'FakeConv', 'ModifiedFakeBlock')

# you must update them in blocks.py as well

# 2. you must import these layers and
# update "parse_model()" function in the tasks.py file to help it construct model from yaml file
# "ultralytics/nn/tasks.py"

# 3. you must modify yaml file according to your desire structure
# "C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\venv\Lib\site-packages\ultralytics\cfg\models\v8\yolov8.yaml"
# You must explicitly specify this yaml path for model when training
####################################################################################################################

# Load the pre-trained DETR model globally(Old)
# DETR_MODEL = DetrModel.from_pretrained('facebook/detr-resnet-50')
# CONFIG = DETR_MODEL.config

# Load the pre-trained model and configuration
model_name = 'facebook/detr-resnet-50'
custom_config = AutoConfig.from_pretrained(model_name)
detr_model = AutoModel.from_pretrained(model_name, config=custom_config)

# You can modify pretrained config by config.variable

class DETREncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Use the global variable to access the encoder
        self.encoder = detr_model.encoder
        self.decoder = detr_model.decoder
        self.config = custom_config
        self.query_position_embeddings = nn.Embedding(self.config.num_queries, self.config.d_model)

    def forward(self, embedding_output):
        flattened_features, flattened_mask, flattened_object_queries, batch_size = embedding_output
        encoder_outputs = self.encoder(inputs_embeds=flattened_features, attention_mask=flattened_mask, object_queries=flattened_object_queries)

        # # Sent query embeddings + object_queries through the decoder (which is conditioned on the encoder output)
        # query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # queries = torch.zeros_like(query_position_embeddings)

        # decoder_outputs = self.decoder(
        #     inputs_embeds=queries,
        #     attention_mask=None,
        #     object_queries=flattened_object_queries,
        #     query_position_embeddings=query_position_embeddings,
        #     encoder_hidden_states=encoder_outputs[0],
        #     encoder_attention_mask=flattened_mask
        # )
        # # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        # return print("Decoder_Outputs: ",decoder_outputs)

        return encoder_outputs.last_hidden_state


class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = custom_config
        self.object_queries = build_position_encoding(self.config)

    def forward(self, x):

        # 2. get the feature map's shape, create pixel mask
        batch_size, channels, height, width = x.shape
        pixel_mask = torch.ones(((batch_size, height, width)))

        # 3. get object query based on the feature map and mask
        feature, object_queries = self.get_queries(x, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = feature

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        # 4. apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        input_projection = nn.Conv2d(channels, self.config.d_model, kernel_size=1)
        projected_feature_map = input_projection(feature_map)

        # 5. flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        flattened_object_queries = object_queries.flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        return flattened_features, flattened_mask, flattened_object_queries, batch_size

    # this method replaces role of DetrConvModel which uses DetrConvEncoder since we already have feature map from SPPF
    def get_queries(self, feature_map: torch.Tensor, pixel_mask: torch.Tensor):
        mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
        out = (feature_map, mask)
        pos = self.object_queries(feature_map, mask).to(feature_map.dtype)
        return out, pos


class GatedFusion(nn.Module):
    def __init__(self, cnn_channels, detr_channels, output_channels):
        super().__init__()
        self.output_channels = output_channels

        # Learnable transformation to reshape the DETR output
        self.detr_to_spatial = nn.Sequential(
            nn.Linear(detr_channels, output_channels * 20 * 20),
            nn.ReLU(),
            nn.Unflatten(1, (output_channels, 20, 20))
        )

        # Transform the CNN feature map
        self.cnn_transform = nn.Conv2d(cnn_channels, output_channels, kernel_size=1)

        # Learnable gate
        self.gate = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feature_map, detr_output):
        # Transform DETR output to a spatial format and reshape
        detr_spatial = self.detr_to_spatial(detr_output.view(-1, 256))
        detr_spatial = detr_spatial.view(-1, self.output_channels, 20, 20)

        # Transform CNN feature map
        cnn_transformed = self.cnn_transform(cnn_feature_map)

        # Concatenate and apply the gate
        combined = torch.cat([detr_spatial, cnn_transformed], dim=1)
        gate_values = self.gate(combined)

        # Apply the gated values
        gated_output = gate_values * cnn_transformed + (1 - gate_values) * detr_spatial

        return gated_output

