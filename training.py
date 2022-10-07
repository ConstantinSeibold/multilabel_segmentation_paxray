from tqdm import tqdm
import torch, os
import numpy as np
import pdb
from utils.init_utils import dict_to_cuda, omit_if_multigpu, set_mode_model, reduce_batch_size

def clip_grad(parameters, args):
    if args.clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)


def train_epoch(model, train_dataloader, loss_fn, optimizer,  writer, epoch, args):
    model.train()
    i=0
    total_loss = 0
    with tqdm(total=len(train_dataloader), desc='Step at start {}; Training epoch {}/{}'.format(i, epoch, args.epochs)) as pbar:
        for i,ret in enumerate(train_dataloader):
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
                print('forward',e)
                import pdb; pdb.set_trace()

            try:
                loss_dict = loss_fn({**out,**pass_input})
            except Exception as e:
                print('loss',e)
                import pdb; pdb.set_trace()

            fin_loss =  loss_dict['loss']
            try:
                optimizer.zero_grad()
                fin_loss.backward()
                clip_grad(model.parameters(),args)
                optimizer.step()
            except Exception as e:
                print('back',e)
                import pdb; pdb.set_trace()


            total_loss += fin_loss.item()

            pbar.set_postfix(loss = '{:.2f},{:.2f}'.format(total_loss/(i+1),fin_loss.item()))
            pbar.update()

            ## Put into Logger Functions
            try:
                writer.update({**out, **loss_dict}, epoch, 1)
            except Exception as e:
                print(e, 'logger ist schuld')
                import pdb; pdb.set_trace()



            if args.store_specific_sample != 'none' and i == 0 and args.store_eval:
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    spec_sample = train_dataloader.dataset.get_specific_sample(args.store_specific_sample)
                    if spec_sample == {}:
                        spec_sample = train_dataloader.dataset.get_specific_sample(train_dataloader.dataset.df.iloc[len(train_dataloader.dataset)-1][0])
                        if spec_sample == {}:
                            continue

                    if len(spec_sample['data'].shape) ==3:
                        spec_sample['data'] = spec_sample['data'].unsqueeze(0)

                    if args.seg_mode =='binary':
                        if len(spec_sample['mask_target'].shape) ==3:
                            spec_sample['mask_target'] = spec_sample['mask_target'].unsqueeze(0)
                    else:
                        if len(spec_sample['mask_target'].shape) ==2:
                            spec_sample['mask_target'] = spec_sample['mask_target'].unsqueeze(0)

                    if len(spec_sample['supervision_type'].shape) == 0:
                        spec_sample['supervision_type'] = spec_sample['supervision_type'].unsqueeze(0)


                    spec_sample['data']       = torch.cat([spec_sample['data'].float(),pass_input['data'].float()],0)
                    spec_sample['mask_target'] = torch.cat([spec_sample['mask_target'].float().view(1,spec_sample['mask_target'].shape[-2],spec_sample['mask_target'].shape[-1]),pass_input['mask_target'].float()],0)
                    spec_sample['supervision_type'] = torch.cat([spec_sample['supervision_type'].view(1), torch.ones(len(pass_input['supervision_type'])).to(spec_sample['data'].device)],0)

                    out = model(spec_sample)
                    os.makedirs(os.path.join(args.log_dir,args.name,'features'),exist_ok=True)
                    np.save(os.path.join(args.log_dir,args.name,'features','features_unlabeled_{}.npy'.format(epoch)), out['segmentation_features'][0].cpu().numpy())
                    np.save(os.path.join(args.log_dir,args.name,'features','features_labeled_{}.npy'.format(epoch)), out['segmentation_features'][-1].cpu().numpy())
                    np.save(os.path.join(args.log_dir,args.name,'features','labels_labeled_{}.npy'.format(epoch)), out['mask_target'][-1].cpu().numpy())
                    # print('sample pred')
                    loss_dict = loss_fn({**out,**spec_sample})
                    writer.store_specific_sample({**out, **loss_dict}, epoch)
