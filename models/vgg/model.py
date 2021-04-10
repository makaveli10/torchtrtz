import os
from torch.nn import Module
import torchvision.models as models
from torchsummary import summary


class VGG16:
    def __init__(self, batch_norm=False) -> None:
        self.model = batch_norm
        self._bn = batch_norm

        # Set model to eval mode
        self._model.cuda()
        self._model.eval()

    @property
    def model(self) -> Module:
        return self._model

    @model.setter
    def model(self, batch_norm: bool) -> None:
        self._model = models.vgg16_bn(
            pretrained=True) if batch_norm else models.vgg16(pretrained=True)

    def print_summary(self) -> None:
        print(summary(self._model, input_size=(3, 224, 224)))


if __name__ == "__main__":
    vgg = VGG16()
    print(type(vgg.model))