import sys
import os
import time

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from dataset import ProbmapDataset
from model_init import init_model
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, train_loader, epoch, optimizer, scheduler, dev):
    model.train()
    losses = AverageMeter()
    criterion = nn.MSELoss(reduction='sum')
    with tqdm(train_loader) as tbar:
        for train_data, probmap in tbar:
            tbar.set_description("epoch {}".format(epoch))
            train_data = train_data.to(torch.float32).to(dev)
            probmap = probmap.to(dev)

            optimizer.zero_grad()
            outputs,_ = model(train_data)

            loss = criterion(outputs.float(), probmap.float())
            losses.update(loss.item(), train_data.shape[0])

            loss.backward()
            optimizer.step()

            tbar.set_postfix(loss="{:.4f}".format(losses.avg), cur_loss="{:.4f}".format(loss))
            del train_data, probmap, outputs, loss

    scheduler.step()

    return losses

def eval_one_epoch(model, val_loader, val_scores, dev):
    model.train()
    all_gt = []
    all_preds = []
    for _ in val_scores:
        all_preds.append([])

    with tqdm(val_loader) as tbar:
        with torch.no_grad():
            for val_data, probmap, gt_num, _ in tbar:
                tbar.set_description("evaluating")
                val_data = val_data.to(torch.float32).to(dev)
                probmap = probmap.to(dev)
                all_gt.append(gt_num.numpy()[0])

                output,_ = model(val_data)
                avg_pooled_pred_probmap = nn.functional.avg_pool2d(output[0], 3, stride=1, padding=1)
                max_pooled_pred_probmap = nn.functional.max_pool2d(avg_pooled_pred_probmap, 3, stride=1, padding=1)
                pred_dotmap = torch.where(avg_pooled_pred_probmap==max_pooled_pred_probmap, avg_pooled_pred_probmap, torch.full_like(output[0], 0))[0]
                for i in range(len(val_scores)):
                    pred_countmap = torch.where(pred_dotmap>=val_scores[i], 1, 0)
                    pred_count = torch.sum(pred_countmap)
                    all_preds[i].append(pred_count.cpu().numpy())

                del val_data, probmap, output

    maes = []
    rmses = []
    for all_pred in all_preds:
        mae = 0
        rmse = 0
        for i in range(len(all_pred)):
            et_count = all_pred[i]
            gt_count = all_gt[i]

            mae += abs(gt_count-et_count)
            rmse += ((gt_count-et_count)*(gt_count-et_count))

        mae = mae/len(all_pred)
        rmse = np.sqrt(rmse/(len(all_pred)))

        maes.append(float('{:.4f}'.format(mae.tolist())))
        rmses.append(float('{:.4f}'.format(rmse.tolist())))

    return maes, rmses

def save_checkpoint(model, info, name = 'test'):
    state = {
            'epoch': info['epoch'],
            'state_dict': model.state_dict(),
            }
    torch.save(state, os.path.join(info['base_root'], name+'.pth.tar'))

def load_checkpoint(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['state_dict'])
    epoch = state_dict['epoch']+1
    return model,epoch

if __name__ == '__main__':
    config = {
              'max_epoch':100,
              'dev':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
              'model_type':'Resnet50',#configs:'Resnet18','Resnet34','Resnet50','Resnet101'
              'lr_start':1e-4,
              'lr_finish':1e-5,
              'val_scores':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55],
              'probmap_type':'mask'#configs:'point','bbox','mask'
              }
    base_dir = sys.path[0]
    train_dataset = ProbmapDataset(train=True, base_dir=os.path.join(base_dir, 'datasets', 'GBISC'), data_split='train.txt', probmap_type=config['probmap_type'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    val_dataset = ProbmapDataset(train=False, base_dir=os.path.join(base_dir, 'datasets', 'GBISC'), data_split='val.txt', probmap_type=config['probmap_type'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = init_model(config['model_type']).to(config['dev'])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, config['lr_start'])
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epoch'], eta_min=config['lr_finish'])

    if not os.path.exists(os.path.join(base_dir,'run','train')):
        os.mkdir(os.path.join(base_dir,'run','train'))
    train_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.mkdir(os.path.join(base_dir,'run','train',train_name))

    outputfile = open(os.path.join(base_dir,'run','train',train_name,"log.txt"), 'w')
    outputfile.close()

    best_maes = []
    best_epochs = []
    for val_score in config['val_scores']:
        best_maes.append(9999)
        best_epochs.append(0)
    for epoch in range(config['max_epoch']):
        outputfile = open(os.path.join(base_dir,'run','train',train_name,"log.txt"), 'a')
        train_loss = train_one_epoch(model, train_loader, epoch, optimizer, schedular, config['dev'])

        text = 'epoch: ' + str(epoch) + ' train_loss: ' + str(train_loss.avg) + ' lr: ' + str(optimizer.state_dict()['param_groups'][0]['lr']) + '\n'
        save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name)},'last')

        val_mae, val_rmse = eval_one_epoch(model, val_loader, config['val_scores'], config['dev'])
        text = text + 'val_mae: ' + str(val_mae) + '\n' + 'val_rmse: ' + str(val_rmse)
        for i in range(len(config['val_scores'])):
            if val_mae[i]<best_maes[i]:
                save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name)},'best_'+str(config['val_scores'][i]))
                best_maes[i] = val_mae[i]
                best_epochs[i] = epoch

        print(text)
        print(text,file=outputfile)
        outputfile.close()

    outputfile = open(os.path.join(base_dir,'run','train',train_name,"log.txt"), 'a')
    text = 'best_val_weights:\n'
    text = text + str(config['val_scores']) + '\n' + str(best_epochs) + '\n' + str(best_maes)
    print(text)
    print(text,file=outputfile)
    outputfile.close()
