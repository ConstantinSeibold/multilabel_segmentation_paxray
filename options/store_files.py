from distutils.dir_util import copy_tree
import os

def store_files(opt):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.isdir(expr_dir):
        os.makedirs(expr_dir)
    with open(opt.name+'.yml','w') as yml:
        for f in args.keys():
            yml.write( '{}: {}\n'.format(f.lower(),args[f]))
