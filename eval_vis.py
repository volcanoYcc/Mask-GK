import sys
import os

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import random
import  matplotlib.pyplot as plt
import time

import pycocotools.mask as mask_util
import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import ProbmapDataset
from model_init import init_model
from train import load_checkpoint

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def eval_one_epoch_vis(base_dir, model, val_loader, split, val_scores, vis_mask_score, mask_threshold, mode, dev):
    model.train()
    all_gt = []
    all_preds = []
    for _ in val_scores:
        all_preds.append([])
    if mode == 'eval':
        mask_results = []
        with open(os.path.join(base_dir,'datasets','Chengdu_berry_COCO','annotations_COCO_'+split+'.json'), 'r') as f:
            dataset_json_anns = json.load(f)
        image_anns = {image.get('id', None): {
            'file_name': image.get('file_name', ''),
            'height': image.get('height', ''),
            'width': image.get('width', ''),
        } for image in dataset_json_anns.get('images', [])}

    time_all = 0
    with tqdm(val_loader) as tbar:
        with torch.no_grad():
            for val_data, probmap, gt_num, img_name in tbar:
                tbar.set_description("evaluating")
                val_data = val_data.to(torch.float32).to(dev)
                probmap = probmap.to(dev)
                all_gt.append(gt_num.numpy()[0])

                time_pred1 = time.time()
                output,_ = model(val_data)
                avg_pooled_pred_probmap = nn.functional.avg_pool2d(output[0], 3, stride=1, padding=1)
                max_pooled_pred_probmap = nn.functional.max_pool2d(avg_pooled_pred_probmap, 3, stride=1, padding=1)
                pred_dotmap = torch.where(avg_pooled_pred_probmap==max_pooled_pred_probmap, avg_pooled_pred_probmap, torch.full_like(output[0], 0))[0]
                time_pred2 = time.time()
                time_all = time_all + time_pred2 - time_pred1
                for i in range(len(val_scores)):
                    pred_countmap = torch.where(pred_dotmap>=val_scores[i], 1, 0)
                    pred_count = torch.sum(pred_countmap)
                    all_preds[i].append(pred_count.cpu().numpy())

                time_pred1 = time.time()
                pred_countmap = torch.where(pred_dotmap>=vis_mask_score, 1, 0)
                nz = torch.nonzero(pred_countmap).cpu().numpy()#(y,x)
                
                max_pooled_object_probmap,indices = nn.functional.max_pool2d(avg_pooled_pred_probmap, 3, stride=1, padding=1, return_indices=True)
                indices_flatten = torch.reshape(indices,(1,indices.shape[1]*indices.shape[2]))[0].cpu().numpy()
                indices = indices[0]
                map_h,map_w = indices.shape
                avg_pooled_pred_probmap = avg_pooled_pred_probmap[0]
                
                peak_indice_list = []
                for y,x in nz:
                    peak_indice_list.append(y*map_w+x)
                
                while True:
                    temp_indices = torch.zeros_like(indices)
                    temp_indices[:,:] = indices
                    indices[:,:] = indices[indices[:,:]//map_w,indices[:,:]%map_w]
                    if (temp_indices == indices).all():
                        break
                
                indices = torch.where(avg_pooled_pred_probmap>=mask_threshold,indices,-1)
                
                time_pred2 = time.time()
                time_all = time_all + time_pred2 - time_pred1

                if mode == 'vis' or mode == 'save':
                    berry_mask = torch.zeros((map_h,map_w,3)).to(torch.uint8).to(dev)
                    for peak_indice in peak_indice_list:
                        berry_mask_temp = torch.where(indices==peak_indice,1,0)
                        berry_mask[:,:,0] = torch.maximum(berry_mask[:,:,0],berry_mask_temp*random.randint(0,255))
                        berry_mask[:,:,1] = torch.maximum(berry_mask[:,:,1],berry_mask_temp*random.randint(0,255))
                        berry_mask[:,:,2] = torch.maximum(berry_mask[:,:,2],berry_mask_temp*random.randint(0,255))
                    
                    print('\ngt:',all_gt[-1],'pred:',np.array(all_preds)[:,-1].tolist())

                    if mode == 'vis':
                        plt.subplot(231)
                        for y,x in nz:
                            plt.plot(x, y, marker='+', color='coral')
                        img_to_draw = np.transpose(val_data[0].cpu().numpy(),(1,2,0))[..., ::-1]*255
                        plt.imshow(img_to_draw.astype(int))
                        plt.subplot(232)
                        plt.imshow(np.transpose(probmap[0].cpu().numpy(),(1,2,0)))
                        plt.subplot(233)
                        plt.imshow(berry_mask.cpu().numpy())
                        plt.subplot(234)
                        berry_mask=berry_mask.cpu().numpy()
                        img_to_draw = np.where(berry_mask!=0,berry_mask*0.6+img_to_draw*0.4,img_to_draw).astype(int)
                        plt.imshow(img_to_draw)
                        plt.subplot(235)
                        plt.imshow(np.transpose(output[0].cpu().numpy(),(1,2,0)))
                        plt.show(block=True)
                    else:
                        img_to_draw = np.transpose(val_data[0].cpu().numpy(),(1,2,0))*255
                        berry_mask=berry_mask.cpu().numpy()
                        img_to_draw = np.where(berry_mask!=0,berry_mask*0.6+img_to_draw*0.4,img_to_draw).astype(int)
                        cv2.imwrite(os.path.join(base_dir,'run','eval','vis',img_name[0]),img_to_draw)

                elif mode == 'eval':
                    for id in image_anns.keys():
                        if image_anns[id]['file_name'] == img_name[0]:
                            img_id = id
                            break
                    height = image_anns[img_id]['height']
                    width = image_anns[img_id]['width']
                    for peak_indice in peak_indice_list:
                        berry_mask_temp = torch.where(indices==peak_indice,1,0)
                        berry_mask_temp = berry_mask_temp.cpu().numpy()
                        berry_mask_temp = cv2.resize(berry_mask_temp.astype(np.uint8), (width,height), interpolation=cv2.INTER_LINEAR)
                        rle = mask_util.encode(np.array(berry_mask_temp[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                        rle['counts'] = rle["counts"].decode("utf-8")
                        mask_results.append({'image_id': img_id, 'category_id': 0, 'segmentation': rle, "score": float(avg_pooled_pred_probmap[peak_indice//map_w][peak_indice%map_w].cpu().numpy())})
                del val_data, probmap, output

    if mode == 'eval':
        max_size = 0
        min_size = 1000000
        with open(os.path.join(base_dir,'datasets','Chengdu_berry_COCO','annotations_COCO_test.json'), 'r') as f:
            dataset = json.load(f)
            annotations = dataset.get('annotations', [])
            for index, annotation in enumerate(annotations):
                bbox = annotation['bbox']
                size = bbox[2]*bbox[3]
                if size<min_size:
                    min_size = size
                if size>max_size:
                    max_size = size
        split1 = (max_size-min_size)/3+min_size
        split2 = (max_size-min_size)*2/3+min_size

        with open (os.path.join(base_dir,'run','eval','seg_result.json'),'w') as f:
            json.dump(mask_results,f)
        base_dir = sys.path[0]
        cocoGt = COCO(os.path.join(base_dir,'datasets','Chengdu_berry_COCO','annotations_COCO_test.json'))
        cocoDt = cocoGt.loadRes(os.path.join(base_dir,'run','eval','seg_result.json'))
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")
        cocoEval.params.maxDets=[1000,1000,1000]
        cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, split1], [split1, split2], [split2, 1e5 ** 2]]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
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

    return maes, rmses, time_all/len(all_preds)

if __name__ == '__main__':
    config = {
              'dev':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
              'model_type':'Resnet50',#configs:'Resnet18','Resnet34','Resnet50','Resnet101'
              'val_scores':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55],
              'checkpoint':'run/paper_weight/best_50_0.2.pth.tar',
              'split':'test',
              'probmap_type':'mask',#configs:'point','bbox','mask'
              'vis_mask_score':0.2,
              'mask_threshold':9e-3,
              'mode':'save'#configs:'vis','save','eval'
              }
    
    base_dir = sys.path[0]

    if not os.path.exists(os.path.join(base_dir,'run','eval','vis')):
        os.mkdir(os.path.join(base_dir,'run','eval','vis'))

    val_dataset = ProbmapDataset(train=False, base_dir=os.path.join(base_dir, 'datasets', 'GBISC'), data_split=config['split']+'.txt', probmap_type=config['probmap_type'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = init_model(config['model_type']).to(config['dev'])
    model,_ = load_checkpoint(model,os.path.join(base_dir,config['checkpoint']))

    val_mae, val_rmse, mean_time = eval_one_epoch_vis(base_dir, model, val_loader, config['split'], config['val_scores'], config['vis_mask_score'], config['mask_threshold'], config['mode'], config['dev'])

    text = 'val_mae: ' + str(val_mae) + ' val_rmse: ' + str(val_rmse) + ' mean_time: ' + str(mean_time)
    print(text)
