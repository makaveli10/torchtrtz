import torch
import os
import torchvision.models as models
from torchvision import transforms as T
from torchsummary import summary
from PIL import Image


class VGG16:
    def __init__(self, batch_norm=False):
        self.__model = models.vgg16_bn(pretrained=True) if batch_norm else models.vgg16(pretrained=True)
        self.__bn = batch_norm
        self.__model.eval()
        self.__model.cuda()

    def __preprocess(self):
        transform = T.Compose([T.Resize(256),
                               T.CenterCrop(224), 
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])
        return transform(image).cuda()

    def print_summary(self):
        print(summary(self.__model, input_size=(3, 224, 224)))

    def save_weights(self, save_path=None):
        if save_path is None:
            save_path = "vgg16_bn.pth" if self.__bn else "vgg16.pth"
        
        if not os.path.exists(save_path):
            torch.save(self.__model, save_path)
            print("Saved Weights: ", save_path)

    def infer(self, image=None):
        if image:
            image = self.__preprocess(image=image)
            image = image.unsqueeze(axis=0)
        else:
            image = torch.ones(1, 3, 224, 224).cuda()
        return self.__model(image)


if __name__=="__main__":
    vgg = VGG16()
    vgg.save_weights()
    vgg.print_summary()
    out = vgg.infer()
    print(out.shape)