import yaml

def get_cfg(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    return cfg

def load_CCNet():

    # CCNet model loading (iris segmentation)
    model = UNet(NUM_CLASSES, NUM_CHANNELS)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    if args.state:
        try:
            if args.cuda:
                model.load_state_dict(torch.load(args.state))
            else:
                model.load_state_dict(torch.load(args.state, map_location=torch.device('cpu')))
                # print("model state loaded")
        except AssertionError:
            print("assertion error")
            model.load_state_dict(torch.load(args.state,
                map_location = lambda storage, loc: storage))
    model.eval()
    softmax = nn.LogSoftmax(dim=1)

    return model, softmax