

def get_loss(cf):
    if cf.losses == '_class':
        from .losses import class_losses
        return class_losses(cf)
    elif cf.losses =='segmentation':
        from .losses import segmentation_losses
        return segmentation_losses(cf)
    elif cf.losses =='pseudolabels_segmentation':
        if cf.additional_rand_aug:
            from .losses import pseudolabels_aug_segmentation_losses
            return pseudolabels_aug_segmentation_losses(cf)
        else:
            from .losses import pseudolabels_segmentation_losses
            return pseudolabels_segmentation_losses(cf)
    elif cf.losses =='online_segmentation':
        from .losses import online_segmentation_losses
        return online_segmentation_losses(cf)
    elif cf.losses =='online_scaled_segmentation':
        from .losses import online_scaled_segmentation_losses
        return online_scaled_segmentation_losses(cf)
    elif cf.losses =='base_segmentation':
        from .losses import base_segmentation_losses
        return base_segmentation_losses(cf)
    elif cf.losses =='reference_segmentation':
        from .losses import reference_segmentation_losses
        return reference_segmentation_losses(cf)
    elif cf.losses =='multiscale_reference_segmentation':
        from .losses import multiscale_reference_segmentation_losses
        return multiscale_reference_segmentation_losses(cf)
    elif cf.losses =='cluster_reference_segmentation':
        from .losses import cluster_reference_segmentation_loss
        return cluster_reference_segmentation_loss(cf)
    elif cf.losses =='distr_reference_segmentation':
        if cf.additional_rand_aug:
            from .losses import distr_aug_reference_segmentation_loss
            return distr_aug_reference_segmentation_loss(cf)
        else:
            from .losses import distr_reference_segmentation_loss
            return distr_reference_segmentation_loss(cf)
    elif cf.task =='segmentation':
        from .losses import segmentation_losses
        return segmentation_losses(cf)
    elif cf.model in ['segmentation_cycle_gan','cut','cycle_gan']:
        return
    raise NameError('{} does not have a fitting LOSS FUNCTION'.format(cf.model))
