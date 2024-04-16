import os
from os.path import join, isdir
from tracker_tracking import *
import numpy as np
import torch
import argparse

import pickle

import math


def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)
    if 'RGBT' in set_type:
        img_list_visible = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        img_list_infrared = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
    elif 'GTOT' in set_type:
        img_list_visible = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        img_list_infrared = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        gt = np.loadtxt(seq_path + '/init.txt')
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
    return img_list_visible,img_list_infrared,gt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'RGBT210' )
    parser.add_argument("-model_path", default = './models/GTOT_ALLv3_982.pth')
    #parser.add_argument("-result_path", default = './results/RGBT234/GTOT_ALL/')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-visualize",default=False, action='store_true')
    parser.add_argument("-adaptive_align",default=True, action='store_false')
    parser.add_argument("-padding",default=1.2, type = float)
    parser.add_argument("-jitter",default=True, action='store_false')
    args = parser.parse_args()
    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    opts['model_path']=args.model_path
    opts['result_path']='./results/'+args.set_type+'/'+args.model_path.split('/')[-1].split('.')[0]+'/'
    opts['visual_log']=args.visual_log
    opts['set_type']=args.set_type
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    print opts


    ## path initialization
    dataset_path = '/DATA/liulei/'

    seq_home = dataset_path + opts['set_type']
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]

    iou_list=[]
    fps_list=dict()
    bb_result = dict()
    result = dict()

    iou_list_nobb=[]
    bb_result_nobb = dict()
    if not os.path.exists(opts['result_path']):
            os.mkdir(opts['result_path'])
    for num,seq in enumerate(seq_list):
        torch.cuda.empty_cache()
        np.random.seed(123)
        torch.manual_seed(456)
        torch.cuda.manual_seed(789)
        if 'txt' in seq or args.model_path.split('/')[-1].split('.')[0]+'_'+seq+'.txt' in os.listdir(opts['result_path']) or num<-1:
            continue
        seq_path = seq_home + '/' + seq
        img_list_v,img_list_i,gt=genConfig(seq_path,opts['set_type'])
        iou_result, result_bb, fps, result_nobb = run_mdnet(img_list_v, img_list_i, gt[0], gt, seq = seq, display=opts['visualize'])
        enable_frameNum = 0.
        for iidx in range(len(iou_result)):
            if (math.isnan(iou_result[iidx])==False): 
                enable_frameNum += 1.
            else:
                ## gt is not alowed
                iou_result[iidx] = 0.

        iou_list.append(iou_result.sum()/enable_frameNum)
        bb_result[seq] = result_bb
        fps_list[seq]=fps

        bb_result_nobb[seq] = result_nobb
        print '{} {} : {} , total mIoU:{}, fps:{}'.format(num,seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list))
        for i in range(len(result_bb)):
            with open(opts['result_path']+args.model_path.split('/')[-1].split('.')[0]+'_'+seq+'.txt', 'a') as f:
                res='{} {} {} {} {} {} {} {}'.format(result_bb[i][0],result_bb[i][1],result_bb[i][0]+result_bb[i][2],result_bb[i][1],result_bb[i][0]+result_bb[i][2],result_bb[i][1]+result_bb[i][3],result_bb[i][0],result_bb[i][1]+result_bb[i][3]) 
                f.write(res)
                f.write('\n')
