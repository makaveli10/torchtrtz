import torch
import os
import torchvision.models as models
from torchvision import transforms as T
from torchinfo import summary
from PIL import Image


class DenseNet:
    def __init__(self):
        self.__model = models.densenet121(pretrained=True)
        self.__model.cuda()
        self.__model.eval()

    def __preprocess(self):
        transform = T.Compose([T.Resize(256),
                               T.CenterCrop(224), 
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])
        return transform(image).cuda()

    def print_summary(self):
        print(summary(self.__model, input_size=(1, 3, 224, 224)))
        print(self.__model)

    def save_weights(self, save_path=None):
        if save_path is None:
            save_path = "densenet121.pth"
        
        if not os.path.exists(save_path):
            torch.save(self.__model, save_path)
            print("Saved Weights: ", save_path)

    def infer(self, image=None):
        with torch.no_grad():
            if image:
                image = self.__preprocess(image=image)
                image = image.unsqueeze(axis=0)
            else:
                image = torch.ones(1, 3, 224, 224).cuda()
            return self.__model(image)


if __name__=="__main__":
    dn = DenseNet()
    dn.save_weights()
    dn.print_summary()
    out = dn.infer()
    print(out[0][:5])