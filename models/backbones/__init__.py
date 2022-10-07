def get_backbone(opt):
    if opt.network in ['resnet_9blocks', 'resnet_6blocks', 'resnet_4blocks', 'unet_128', 'unet_256', 'stylegan2', 'smallstylegan2', 'resnet_cat',]:
        from .backbone_translation import define_G
        return define_G(opt.input_nc, opt.classes, opt.ngf, opt.network)
    elif opt.network in ['basic','n_layers','pixel','stylegan2_dis',]:
        from .backbone_translation import define_D
        return define_D(opt.input_nc, opt.ndf, opt.network)
    elif not 'fpn' in opt.network:
        from .backbone import backbone
        return backbone(opt)
    else:
        from .backbone_fpn import resnet_fpn_backbone
        opt.network = opt.network[3:]
        return resnet_fpn_backbone(opt.network, pretrained=opt.pt_pretrained, out_channels = opt.ngf,trainable_layers=3 if not opt.freeze_backbone else 0)
