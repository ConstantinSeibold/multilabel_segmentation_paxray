import torch,numpy as np
import time
import numpy as np; np.random.seed(0)
import warnings
import pdb
from options.opt import Options
# from utils.eval_utils import *
from utils.init_utils import *


def main(args):
    model = get_model(args)
    model = load_model(model,args)
    
    test_loader = get_dataloader(args)
    test_loader.dataset.change_mode(get_dataset_eval_mode(args,'test'))

    for i,ret in enumerate(test_loader):
        torch.cuda.empty_cache()
        
        
        try:
            pass_input = dict_to_cuda(ret)
        except Exception as e:
            print('cuda',e)
            import pdb; pdb.set_trace()

        try:
            pass_input_ = omit_if_multigpu(pass_input, args)
        except Exception as e:
            print('omit',e)
            import pdb; pdb.set_trace()
        try:
            out = model(pass_input)
        except Exception as e:
            print('omit',e)
            import pdb; pdb.set_trace()

        os.makedirs(args.pred_folder, exist_ok=True)
        for j in range(len(ret['filename'])):
            pred_to_store = torch.nn.functional.interpolate(out['segmentation_preds'][j].unsqueeze(0).float(), (out['orig_size'][j][0].cpu().item(),out['orig_size'][j][1].cpu().item()), mode='nearest').cpu().numpy().astype(np.bool)[0]
            np.save(os.path.join(args.pred_folder, ret['filename'][j][:-4]+'.npy'), pred_to_store)
                   

if __name__ == '__main__':
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(23)
    warnings.filterwarnings("ignore")
    args = Options().parse()

    main(args)
