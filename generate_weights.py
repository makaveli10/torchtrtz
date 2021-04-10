"""Python file to generate wts format of weights from torch models

Returns:
    weights.wts: Saves the weights of the model.
"""
import argparse
import struct
import torch


class Model:
    """Load models based on the command line argument name provided.
    """
    def __init__(self, opt: argparse.Namespace) -> None:
        """Initialize model

        Args:
            opt (argparse.Namespace): Command line arguments provided.
        """
        self.model = opt.model

    @property
    def model(self) -> torch.nn.Module:
        """Getter for the model

        Returns:
            Module: torch model
        """
        return self._model

    @model.setter
    def model(self, model_name: str) -> None:
        """Setter for the model. Loads model based on the name provided in command-line arguments.

        Args:
            model_name (str): Model name provided.
        """
        if model_name == "VGG16":
            from models.vgg import VGG16
            self._model = VGG16().model
        elif model_name == "DenseNet121":
            from models.densenet import DenseNet
            self._model = DenseNet().model
        elif model_name == "AlexNet":
            from models.alexnet import AlexNet
            self._model = AlexNet().model
        elif model_name == "Inceptionv4":
            from models.inception_v4 import InceptionV4
            self._model = InceptionV4().model

    def generate_weights(self, opt: argparse.Namespace) -> None:
        """Convert torch weights format to wts weights format

        Args:
            opt (argparse.Namespace): Command line arguments provided.
            model (torch.nn.Module): Model based on model name provided.
        """
        # open tensorrt weights file
        wts_file = open(opt.save_trt_weights, "w")

        # write length of keys
        print("Keys: ", self._model.state_dict().keys())
        wts_file.write("{}\n".format(len(self._model.state_dict().keys())))
        for key, val in self._model.state_dict().items():
            print("Key: {}, Val: {}".format(key, val.shape))
            vval = val.reshape(-1).cpu().numpy()
            wts_file.write("{} {}".format(key, len(vval)))
            for v_l in vval:
                wts_file.write(" ")

                # struct.pack Returns a bytes object containing the values v1, v2, …
                # packed according to the format string format (>big endian in this case).
                wts_file.write(struct.pack(">f", float(v_l)).hex())
            wts_file.write("\n")

        wts_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        help=
        'state the model name (along with layer information) based on the README file'
    )
    parser.add_argument('--save-trt-weights',
                        type=str,
                        default='vgg16.wts',
                        help='save path for TensorRT weights')
    args = parser.parse_args()

    dl_model = Model(args)
    dl_model.generate_weights(args)
