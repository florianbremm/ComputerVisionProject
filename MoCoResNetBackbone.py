import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

# Load the 800-epoch MoCo v2 checkpoint
_checkpoint_url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
_checkpoint = torch.hub.load_state_dict_from_url(_checkpoint_url, map_location="cpu")

# Define a standard ResNet-50 backbone
_resnet50 = models.resnet50()
_encoder = nn.Sequential(*list(_resnet50.children())[:-1])  # Remove final FC layer

# Load MoCo weights into the model (encoder_q is the online encoder)
_state_dict = _checkpoint['state_dict']
_new_state_dict = {}

for k, v in _state_dict.items():
    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        new_k = k.replace('module.encoder_q.', '')
        _new_state_dict[new_k] = v

_msg = _encoder.load_state_dict(_new_state_dict, strict=False)
print("Loaded keys:", _msg)

# Wrap in a model that outputs flattened features
class MoCoResNetBackbone(nn.Module):
    def __init__(self, encoder=_encoder):
        super().__init__()
        self.encoder = encoder  # Output: (B, 2048, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        return torch.flatten(x, 1)  # Output: (B, 2048)

# Instantiate final model