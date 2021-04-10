"""Load VGG16 model and return it.

Returns:
    Module: Torch Module of VGG16.
"""
from torch.nn import Module
from torchsummary import summary
import torchvision.models as models


class VGG16:
    """Loads VGG16 model.
    """
    def __init__(self, batch_norm=False) -> None:
        """Initialize model based on batch normalization parameter.

        Args:
            batch_norm (bool, optional): Include batch normalization layer. Defaults to False.
        """
        self._model = models.vgg16_bn(
            pretrained=True) if batch_norm else models.vgg16(pretrained=True)
        self._bn = batch_norm

        # Set model to eval mode
        self._model.cuda()
        self._model.eval()

    @property
    def model(self) -> Module:
        """Getter for the model

        Returns:
            Module: torch model
        """
        return self._model

    def print_summary(self) -> None:
        """Print summary of the model.
        """
        print(summary(self._model, input_size=(3, 224, 224)))


if __name__ == "__main__":
    vgg = VGG16()
    print(type(vgg.model))
