import argparse
import struct
import torch

def generate_weights(opt):
    if not opt.weights:
        print("Please provide vgg torch weights file")
        return

    # Load model
    model = torch.load(opt.weights)
    model = model.cuda()
    model = model.eval()

    # open tensorrt weights file
    f = open(opt.save_trt_weights, "w")

    # write length of keys
    print("Keys: ", model.state_dict().keys())
    f.write("{}\n".format(len(model.state_dict().keys())))
    for key, val in model.state_dict().items():
        print("Key: {}, Val: {}".format(key, val.shape))
        vval = val.reshape(-1).cpu().numpy()
        f.write("{} {}".format(key, len(vval)))
        for v in vval:
            f.write(" ")

            # struct.pack Returns a bytes object containing the values v1, v2, … 
            # packed according to the format string format (>big endian in this case).
            f.write(struct.pack(">f", float(v)).hex())
        f.write("\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='vgg16.pth', help='pytorch weights model.pth path(s)')
    parser.add_argument('--save-trt-weights', type=str, default='vgg16.wts', help='save path for tensorrt weights')
    opt = parser.parse_args()
    print(opt)
    generate_weights(opt)