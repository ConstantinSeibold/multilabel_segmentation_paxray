import torch
import time
import numpy as np; np.random.seed(0)
import warnings

from options.opt import Options as TrainOptions
# from utils.eval_utils import *
from utils.init_utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):

    writer = get_logger(args)
    print('Logger initialized')

    init = args.train

    ## DATASETS
    train_loader = get_dataloader(args)
    args.classes = len(train_loader.dataset.classes)

    print('{}-Loader initialized with {} images.'.format(args.mode, train_loader.dataset.__len__()))
    orig_batch_size = args.batch_size
    args.mode = get_dataset_eval_mode(args,'val')
    args.batch_size = 1
    val_loader = get_dataloader(args)
    val_loader.dataset.change_mode(get_dataset_eval_mode(args,'val'))
    print('{}-Loader initialized with {} images.'.format(args.mode, val_loader.dataset.__len__()))

    args.mode = get_dataset_eval_mode(args,'test')
    args.batch_size = 1
    test_loader = get_dataloader(args)
    test_loader.dataset.change_mode(get_dataset_eval_mode(args,'test'))
    print('{}-Loader initialized with {} images.'.format(args.mode, test_loader.dataset.__len__()))

    train_epoch = get_training_procedure(args)
    print('Gathered training procedure')
    evaluator = get_eval(args)
    print('Initialized Evaluation Module')
    args.start_epoch = 1

    model = get_model(args)
    optimizer = set_optimizer(args, model)
    print('Optimizer Initialized')

    model,optimizer = resume(model,optimizer,args)

    scheduler = get_scheduler(optimizer,args)
    print('Scheduler Initialized')

    loss_fn = get_loss(args)
    print('Gathered Loss function')
    args.mode = init
    epoch = 0
    init_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        torch.cuda.empty_cache()
        train_epoch(model, train_loader, loss_fn, optimizer,  writer, epoch, args)

        state = reduce_state(args, model, optimizer, epoch)
        if (epoch+1) % args.val_freq == 0:
            ## Put into Logger and Eval branch
            eval_metrics_val = evaluator.evaluate(model, val_loader)
            torch.cuda.empty_cache()
            print('Target Metric Value: ',eval_metrics_val['metric_target'])
            writer.add_results(eval_metrics_val,'val', epoch)
            writer.log_results(eval_metrics_val,'val',test_loader.dataset.get_classes())
            writer.store_if_best(state, eval_metrics_val, epoch)

        if epoch % args.save_freq == 0:
            writer.store(state, epoch)

        scheduler.step()

    used_seconds = time.time()-init_time
    used_hours = used_seconds//3600
    used_minutes = (used_seconds%3600)//60
    used_seconds = used_seconds%60
    print('Training took {} hours, {} minutes and {} seconds.'.format(used_hours, used_minutes, used_seconds))
    del model
    model = load_best(get_model(args),args)
    torch.cuda.empty_cache()
    eval_metrics_test = evaluator.evaluate(model, test_loader)
    print('Test Target Metric Value: ',eval_metrics_test['metric_target'])
    writer.add_results(eval_metrics_test, 'test', epoch)
    writer.log_results(eval_metrics_test,'test', test_loader.dataset.get_classes())


if __name__ == '__main__':
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(23)
    warnings.filterwarnings("ignore")
    args = TrainOptions().parse()

    main(args)
