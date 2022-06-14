import torch
from torchvision.transforms.functional import rotate
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform

@register_transform("DiscreteRotation")
class DiscreteRotation(ClassyTransform):
    """Rotate image by one of the given angles.

    Arguments:
        angles: list(ints). List of integer degrees to pick from. E.g. [0, 90, 180, 270] for a random 90-degree-like rotation
    """

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = self.angles[torch.randperm(len(self.angles))[0]]
        return rotate(x, angle)

    def __repr__(self):
        return f"{self.__class__.__name__}(angles={self.angles})"

    @classmethod
    def from_config(cls, config):
        """
        Instantiates DiscreteRotation from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            DiscreteRotation instance.
        """
        print("config", config)
        angles = config.get("angles", [0, 90, 180, 270])
        return cls(angles=angles)