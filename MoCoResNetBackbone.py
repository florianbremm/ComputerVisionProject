import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

# Load the 800-epoch MoCo v2 checkpoint
_checkpoint_url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
_checkpoint = torch.hub.load_state_dict_from_url(_checkpoint_url, map_location="cpu")

# Load MoCo weights into the model (encoder_q is the online encoder)
_state_dict = _checkpoint['state_dict']
_new_state_dict = {}

for k, v in _state_dict.items():
    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        new_k = k.replace('module.encoder_q.', '')
        _new_state_dict[new_k] = v

# Wrap in a model that outputs flattened features
class MoCoResNetBackbone(nn.Module):
    def __init__(self, dim=2048, momentum=0.999):
        super().__init__()
        self.momentum = momentum

        # Create the online (query) encoder
        self.encoder_q = _build_encoder(dim)

        # Create the momentum (key) encoder
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
        Input:
            im_q: query image
            im_k: key image (optional)
        Output:
            q, k: query and key representations
        """
        q = self.encoder_q(im_q)
        if im_k is None:
            return q
        with torch.no_grad():
            if momentum_update:
                self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
        return q, k

def _build_encoder(dim):
    # Define a standard ResNet-50 backbone
    resnet50 = models.resnet50()
    num_features = resnet50.fc.in_features
    encoder = nn.Sequential(*list(resnet50.children())[:-1])  # Remove final FC layer
    msg = encoder.load_state_dict(_new_state_dict, strict=False)
    projection = nn.Linear(num_features, dim)
    return nn.Sequential(encoder, nn.Flatten(), projection)