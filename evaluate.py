import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

import time
import os
import h5py
import logging

from utils import timeSince, TimeChecker, MyDataParallel
from data_loader import ClipFeatureDataSet, VideoData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_features(save_dir, count, ft_dic):
    nfs = 0
    h5f_name = 'extracted_features_%d.h5' % count
    h5f_path = os.path.join(save_dir, h5f_name)
    h5f = h5py.File(h5f_path , 'w' )
    for vname,features in ft_dic.items():
        nfs += features.shape[0]
        h5f.create_dataset(vname, data = features, compression="gzip")
    h5f.close()
    print("Total %d features are saved into %s" % (nfs, h5f_path))
    return h5f_name


def delete_h5_files(ft_dir):
    for file_ in os.listdir(ft_dir):
        if file_.endswith(".h5"):
            os.remove(os.path.join(ft_dir, file_))


def extract_feature(cnn_model, tf, video_root, v_list_file, ft_dir):
    """
    Extract spatial features for each video frame with a backbone cnn model and save features to h5py files
    """
    dataset = VideoData(video_root, v_list_file, tf)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=dataset.video_unit_batch_sampler(), num_workers=8, pin_memory = True)

    ft_dic = {}
    max_frames = 50000
    count = 0
    frames = 0

    print( "extraction start" )
    start = time.time()
    cnn_model.eval()
    org_mode = cnn_model.extract_feature_mode(True)
    
    with torch.no_grad():
        for (images, _), vname  in zip(dataloader, dataset.video_names()):
            features = cnn_model(images.to(device)).cpu().numpy()
            ft_dic[vname] = features
            frames += features.shape[0] 

            if frames > max_frames:
                save_features(ft_dir, count, ft_dic)
                print( timeSince(start) )
                frames = 0
                count += 1
                ft_dic = {}

    cnn_model.extract_feature_mode(org_mode)

    if len(ft_dic) > 0:
        save_features(ft_dir, count, ft_dic)
        print( timeSince(start) )

def feature_to_loss_lrcn(model, label, feature, criterion): #feature: T(B, T, feature_size), label: T()
    outputs = model.forward_lstm(feature)#.forward_from_static_features( feature ) #T(B, T, A)
    outputs = outputs.view(-1, model.action_size) #T(B*T, A)

    expanded_label = torch.LongTensor([label] * outputs.size(0)).to(device)#torch.cat( [torch.LongTensor(label)] * outputs.size(0) ).to(device) #T(B*T)

    loss = criterion( outputs, expanded_label ).item() * outputs.size(0)
    return loss, outputs

def feature_to_loss_1dcnn(model, label, feature, criterion): #feature: T(B, T, feature_size), label: T()
    outputs = model.process_feature(feature) #T(B, A)
    expanded_label = torch.LongTensor([label] * outputs.size(0)).to(device) #T(B)
    loss = criterion( outputs, expanded_label ).item() * outputs.size(0)

    return loss, outputs

def evaluate_model(logger, model_name, model, criterion, stride, ft_dir, video_dir, transform, video_list, device):
    with open(video_list, 'r') as f:
        for line in f:
            line = line.strip()
            class_str, label = line.split()
            label = int(label)
            class_str = class_str.split('/')[0]

    timer = TimeChecker()

    ### Wrap up CNN encoder if the parallel execution is used
    if torch.cuda.device_count() > 1:
        cnn = MyDataParallel(model.cnn)
    else:
        cnn = model.cnn
    cnn.to(device)

    ### Clean feature files generated from the previous evaluation
    delete_h5_files(ft_dir)


    ### Feature extraction
    timer.start()
    logger.info("Feature extreaction starts")
    extract_feature(cnn, transform, video_dir, video_list, ft_dir)
    logger.info("Feature extreaction ends elapsed %s" % timer.elapsed())

    ### Construct feature loader
    featureLoader = ClipFeatureDataSet(ft_dir, video_list, seq_len = model.seq_len, stride = stride, batch_size=32).generator()
    

    if model_name == "lrcn":
        feature_to_loss = feature_to_loss_lrcn
    elif model_name == "1dcnn":
        feature_to_loss = feature_to_loss_1dcnn


    prev_label = None
    prev_vname = None
    total_frames_in_clips_by_video = 0
    aggregated_score = 0

    total_frames_in_clips = 0
    nvideo = 0
    correct = 0
    loss = 0

    model.eval()
    timer.start()
    logger.info("Evaluation Starts")
    with torch.no_grad():
        ### 'featureLoader' returns features for unit video clip at every interation 
        ### e.g. 1st iteration) it returns features for Video0[0, ... , seq_len - 1]
        ###      2nd iteration) it returns features for Video0[stride, ... , seq_len - 1 , stride]
        for feature, vname, label, _debug_indices in featureLoader: # feature: T(B, T, feature_size)
            
            if prev_vname is None:
                ### It's entered only at the first iteration
                prev_label = label
                prev_vname = vname

            feature = feature.to(device)

            loss_, outputs = feature_to_loss(model, label, feature, criterion)

            loss += loss_

            x = F.softmax( outputs, dim = 1 ) # T(B*T, A) or # T(B, A)

            total_frames_in_clips += feature.size(0) * feature.size(1) #x.size(0)

            if prev_vname != vname:
                ### Features for a new video are started being loaded
                ### Make the final prediction for the previous video
                val, idx = torch.max(aggregated_score, dim = 0)
                if prev_label == idx.item():
                    correct += 1

                nvideo += 1
                total_frames_in_clips_by_video = feature.size(0) * feature.size(1) #x.size(0)

                aggregated_score = torch.sum( x , dim = 0 )
                prev_label = label
                prev_vname = vname
            else:
                ### Features for the current video are being loaded
                ### Accumulate a prediction score for the final prediction
                total_frames_in_clips_by_video += feature.size(0) * feature.size(1) #outputs.size(0)
                aggregated_score += torch.sum( x , dim = 0 ) # T(A)

        if isinstance(aggregated_score, torch.Tensor):
            val, idx = torch.max(aggregated_score, dim = 0)
            if label == idx.item():
                correct += 1

            total_frames_in_clips += total_frames_in_clips_by_video
            nvideo += 1

    logger.info("Evaluation Ends %s elapsed" % timer.elapsed())
    accuracy = correct / nvideo
    loss /= total_frames_in_clips

    logger.info("Valid Result for %d videos and %d frames in clips" % (nvideo, total_frames_in_clips))
    logger.info("Accuracy - %.3f, Loss - %.3f" % (accuracy, loss))

    delete_h5_files(ft_dir)

    return accuracy, loss
