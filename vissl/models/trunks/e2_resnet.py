import torch
import torch.nn as nn
from vissl.models.trunks import register_model_trunk

from vissl.models.model_helpers import Flatten
from typing import List
from vissl.config import AttrDict
import e2cnn

# temporary hack to import rissl
# TODO: Improve it once the final directory structure is determined
import sys
sys.path.insert(0, "/home/b.dolicki/thesis/")
import rissl

# For more depths, add the block config here
BLOCK_CONFIG = {
    18: {
            "block": rissl.models.e2_resnet.E2BasicBlock,
            "layers": (2, 2, 2, 2)
    },
    50: {
            "block": rissl.models.e2_resnet.E2Bottleneck,
            "layers": (3, 4, 6, 3)
    }
}

@register_model_trunk("e2_resnet")
class E2ResNet(nn.Module):
    """
    Generic implementation of equivariant ResNet-like models including WideResNet and ResNext
    based on https://github.com/blazejdolicki/ssl-histo/blob/equivariant-resnet/models/e2_resnet.py

    Required config
        config.MODEL.TRUNK.NAME=e2_resnet
        TODO: Figure out what model parameters we will accept
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(E2ResNet, self).__init__()
        self.model_config = model_config

        # get the params trunk takes from the config
        # FIXME: I think new versions of VISSL need the parameter TRUNK.TRUNK_PARAMS instead of TRUNK,
        # not sure if that's backwards compatible
        self.trunk_config = self.model_config.TRUNK.E2_RESNETS

        # ResNet parameters
        self.depth = self.trunk_config.DEPTH
        self.groups = self.trunk_config.GROUPS
        self.width_per_group = self.trunk_config.WIDTH_PER_GROUP
        self.zero_init_residual = self.trunk_config.ZERO_INIT_RESIDUAL

        # Equivariant parameters
        equivariant_args = {"N": self.trunk_config.N,
                            "F": self.trunk_config.F,
                            "sigma": self.trunk_config.sigma,
                            "restrict": self.trunk_config.restrict,
                            "flip": self.trunk_config.flip,
                            "fixparams": self.trunk_config.fixparams,
                            "conv2triv":self.trunk_config.conv2triv,
                            "deltaorth": self.trunk_config.deltaorth,
                            "last_hid_dims":self.trunk_config.last_hid_dims}

        # Current implementation only supports ResNet50 and ResNext50, to add other models add arguments for
        # `block` and `layers` here and in config.
        model = rissl.models.e2_resnet.E2ResNet(block=BLOCK_CONFIG[self.depth]['block'],
                                                layers=BLOCK_CONFIG[self.depth]['layers'],
                                                groups=self.groups,
                                                width_per_group=self.width_per_group,
                                                zero_init_residual=self.zero_init_residual,
                                                **equivariant_args)

        self.in_lifting_type = model.in_lifting_type

        print(f"Model depth: {self.depth}, layers: {BLOCK_CONFIG[self.depth]}")

        self._feature_blocks = nn.ModuleDict(
            [
                ("conv1", model.conv1),
                ("bn1", model.bn1),
                ("conv1_relu", model.relu),
                ("maxpool", model.maxpool),
                ("layer1", model.layer1),
                ("layer2", model.layer2),
                ("layer3", model.layer3),
                ("layer4", model.layer4),
                # ("grouppool", model.mp),
                ("avgpool", model.avgpool),
                ("flatten", Flatten(1)),
            ]
        )

    def forward(self, x: torch.Tensor, out_feat_keys: List[str] = []) -> List[torch.Tensor]:
        # See the forward pass of resnext.py for reference of how additional features
        # can be implemented. For now, we do not require these advanced features.
        if len(out_feat_keys) > 0:
            raise NotImplementedError

        # wrap input into geometric tensor with the initial representation type
        x = e2cnn.nn.GeometricTensor(x, self.in_lifting_type)

        for i, (feature_name, feature_block) in enumerate(self._feature_blocks.items()):
            if feature_name == "avgpool":
                x = x.tensor
            x = feature_block(x)

        # VISSL expects a list. It either contains one vector (the output), or
        # a list of requested intermediate features
        # For now, we only implement the output of the model.
        return [x]
