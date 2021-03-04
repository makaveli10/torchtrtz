import torch
import os
import torchvision.models as models
from torchvision import transforms as T
from torchsummary import summary
from PIL import Image

class AlexNet:
    def __init__(self):
        self.__model = models.alexnet(pretrained=True)
        self.__model.eval()
        self.__model.cuda()

    def __preprocess(self, image):
        transform = T.Compose([T.Resize(256),
                               T.CenterCrop(224), 
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])
        return transform(image).cuda()

    def print_summary(self):
        print(summary(self.__model, input_size=(3, 224, 224)))

    
    def save_weights(self, save_path="alexnet.pth"):
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
    alex = AlexNet()
    # image = Image.open("zidane.jpg")
    out = alex.infer()
    print(out[0], out[0].shape)
    print(alex.print_summary())
    alex.save_weights()
