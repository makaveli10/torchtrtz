import torch
import struct
import argparse
# from torchsummary import summary


def generate_weights(opt):
    if not opt.weights:
        return 
    
    # Load model from weights path
    net = torch.load(opt.weights)
    net = net.cuda()
    net = net.eval()
    # print(summary(net, (3, 224, 224)))
    
    # open the trt wts file to write weights
    f = open(opt.save_trt_weights,"w")

    # write length of the state_dict keys
    print("Keys")
    print(net.state_dict().keys())
    f.write("{}\n".format(len(net.state_dict().keys())))

    # write weights in contiguous 1d array format
    for key, val in net.state_dict().items():
        print("Key: {}, Value: {}".format(key, val.shape))
        vval = val.reshape(-1).cpu().numpy()
        f.write("{} {}".format(key, len(vval)))
        for v in vval:
            f.write(" ")

            # struct.pack Returns a bytes object containing the values v1, v2, … 
            # packed according to the format string format (>big endian in this case).
            f.write(struct.pack('>f', float(v)).hex())
        f.write("\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='alexnet.pth', help='pytorch weights model.pth path(s)')
    parser.add_argument('--save-trt-weights', type=str, default='alexnet.wts', help='save path for tensorrt weights')
    opt = parser.parse_args()
    print(opt)
    generate_weights(opt)
