def select_dataset(args):
    if args.dataset == 'json':
        from data.general_json_dataset import GenDataset
        return GenDataset(args)
    elif args.dataset == 'folder':
        from data.folder_dataset import FolderDataset
        return FolderDataset(args)
    raise NameError('{} is not an implemented DATASET'.format(args.dataset))
