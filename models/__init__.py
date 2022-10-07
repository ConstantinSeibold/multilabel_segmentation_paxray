def get_model(args):
    # TODO
    # dynamic model load
    if  args.model == 'unet':
        from models.unet import UNet
        return UNet(args)
    elif  args.model == 'backbone_unet':
        from models.unet import BackboneUNet
        return BackboneUNet(args)
    elif args.model == 'segfpn':
        from models.seg_fpn import SegFPNModel
        return SegFPNModel(args)
    raise NameError('{} is not an implemented MODEL'.format(args.model))
