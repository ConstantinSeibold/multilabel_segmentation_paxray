import yaml,os
from argparse import Namespace

def load_yml(path):
    yml = yaml.load(open(path))
    yml = {y.lower():yml[y] for y in yml}
    namespace = Namespace(**yml)
    return namespace

def write_yml(opt):
    args = vars(opt)
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)
    with open(opt.log_dir+'/'+opt.name+'.yml','w') as yml:
        for f in args.keys():
            yml.write( '{}: {}\n'.format(f.lower(),args[f]))
