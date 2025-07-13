import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

# Load the 800-epoch MoCo v2 checkpoint
_checkpoint_url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
_checkpoint = torch.hub.load_state_dict_from_url(_checkpoint_url, map_location="cpu")
_state_dict = _checkpoint['state_dict']


# Remove the prefix from the query encoder keys
_new_state_dict = {}
for k, v in _state_dict.items():
    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        new_k = k.replace('module.encoder_q.', '')
        _new_state_dict[new_k] = v

# Wrap in a model that outputs flattened features
class MoCoResNetBackbone(nn.Module):
    """
    class for the ResNet-50 Backbone
    """
    def __init__(self, dim=2048, momentum=0.999):
        super().__init__()
        self.momentum = momentum

        # Create the query encoder
        self.encoder_q = _build_encoder(dim)

        # Create the key encoder
        self.encoder_k = _build_encoder(dim)

        # Initialize encoder_k with encoder_q weights
        self._momentum_update_key_encoder(0)  # full copy initially

        # Freeze encoder_k gradients
        for param in self.encoder_k.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m=None):
        """
        Update the key encoder using exponential moving average of query encoder.
        """
        if m is None:
            m = self.momentum
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def forward(self, im_q, im_k=None, momentum_update=True):
        """
        forward one or two images through the backbone.
        :param im_q: Image for the query encoder
        :param im_k: (optional) Image for the key encoder
        :param momentum_update: whether to update the key encoder
        :return: query and (optional) key embedding
        """
        q = self.encoder_q(im_q)
        # this methode could be used with only one image
        if im_k is None:
            return q

        # handle the forwarding through the key encoder and updating it
        with torch.no_grad():
            if momentum_update:
                self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
        return q, k
    
    def encode_query(self, x):
        """
        forward one image through the backbone without projection head (projection head only needed in contrastive training).
        :param x: Image to encode
        :return: 2048 dimensional embedding of the Image
        """
        return self.encoder_q[:-1](x)

def _build_encoder(dim):
    """
    Build a ResNet-50 encoder.
    :param dim: output dimension
    :return: a ResNet-50 encoder with projection head
    """

    #load resnet structure
    resnet50 = models.resnet50()
    num_features = resnet50.fc.in_features

    #remove old projection head and add new head in the correct dimension
    encoder = nn.Sequential(*list(resnet50.children())[:-1])  # Remove final FC layer
    projection = nn.Linear(num_features, dim)
    return nn.Sequential(encoder, nn.Flatten(), projection)