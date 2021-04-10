"""Load AlexNet model and return it.

Returns:
    Module: Torch Module of AlexNet.
"""
from torch.nn import Module
from torchsummary import summary
from torchvision.models import alexnet


class AlexNet:
    """Loads AlexNet model.
    """
    def __init__(self) -> None:
        """Initialize AlexNet model.
        """
        self._model = alexnet(pretrained=True)

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
    alex = AlexNet()
    print(alex.print_summary())
