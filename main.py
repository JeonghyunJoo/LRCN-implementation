import numpy as np
import torch
import torch.nn as nn
import torch.optim 
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
from datetime import datetime
import os
import argparse
import logging

from data_loader import TrainVideoDataSet
from utils import TimeChecker, MyDataParallel
from model import LRCN, ODCNN
from evaluate import evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str, required=True) # Directory where video frame data locate
parser.add_argument("-model", type=str, default='lrcn') # Model type: 'lrcn' or '1dcnn'
parser.add_argument("-b", type=int, default=24) # Batchsize 24 batches
parser.add_argument("-dim", type=int, default=256) # Temporal feature dimension
parser.add_argument("-a", type=float, default = 0.1) # Degree of label smoothing
parser.add_argument("-iter", type=int, default = 15) # Number of iterations of training processes
parser.add_argument("-optim", type=str, default = 'sgd') # Optimizer type
parser.add_argument("-lr", type=float, default = 0.001) # Learning rate
parser.add_argument("-eval_since", type=int, default=1) # Start to evaluate after 'eval_since' iterations
parser.add_argument("-eval_interval", type=int, default=5) # Evaluation interval
parser.add_argument("-tsize", type=int, default=4) # Temporal kernel size for 1d CNN
parser.add_argument("-seq", type=int, default=16) # Unit video clip length
parser.add_argument("-weight_file", type = str)

args = parser.parse_args()

# Make a base directory to store expeirment results unless it exists already
if not os.path.exists('experiments'):
    os.mkdir('experiments')

def timestamp_str():
    # Return a current timestamp string
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H_%M_%S_%f")[:-3]
    
    return formatted_time

### Create experiment directory
expr_id = timestamp_str() # Use the timestamp value as an experiment id
expr_dir = os.path.join('experiments', expr_id) # Experiment directory
if not os.path.exists(expr_dir):
    os.mkdir(expr_dir)


### Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

### Log file configuration
formatter = '[%(asctime)s] [%(name)s] [%(levelname)s]:: %(message)s'
logger = logging.getLogger(expr_id)
handler = logging.StreamHandler()
handler.setFormatter( logging.Formatter( formatter ) )
logger.addHandler( handler )
logger.setLevel(logging.INFO)

filehandler = logging.FileHandler(os.path.join(expr_dir, 'log.log'))
filehandler.setFormatter( logging.Formatter( formatter ) )
logger.addHandler( filehandler )

globalTimer = TimeChecker()




class SmoothCrossEntropyLoss():
    """
    CrossEntropyLoss with Label Smoothing
    Instead of one-hot-vector, it distributes some amount of probability mass, which is specified by alpha, over all classes

    Arguments:
     - alpha: alpha value
     - num_cls: total number of classes (e.g. 101 for UCF101)
     - reduction: reduction parameter for the pytorch crossentropy loss
    """
    def __init__(self, alpha, num_cls, reduction = 'mean'):
        self.alpha = alpha
        self.num_cls = num_cls
        self.reduction = reduction
    
    
    def __call__(self, pred, label): #pred; T(N, num_cls) , label: T(N)
        smoothed_dist = torch.zeros_like(pred, device = label.device) #T(N, num_cls)
        smoothed_dist.fill_(self.alpha / self.num_cls) #Spread probability mass over all classes
        smoothed_dist.scatter_(dim = 1, index = label.unsqueeze(1), value = 1 - self.alpha + self.alpha / self.num_cls) #Fill remaining probability to a label class
        
        pred = F.log_softmax(pred, dim = 1) #T(N, num_cls)
        loss = torch.sum( -pred * smoothed_dist, dim = 1) #T(N)
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
        
        return loss



def train_step_1dcnn(model, batch_data, optimizer, criterion):
    '''
    Train '1dcnn' model for one step
    
    Arguments:
    - model: pytorch module object
    - batch_data: python dictionary of {'data' : video clip tensor, 'label' : video label tensor}
    - optimizer: pytorch optimizer object
    - criterion: pytorch loss function object 

    Returns:
    - loss: float, loss value
    - correct: int, the number of correctly predicted videos
    '''
    model.train()
    
    inputs = batch_data['data'] # T(Bn, seq_len, *)
    labels = batch_data['label'].view(-1) # T(Bn)

    inputs = inputs.view(-1, *model.cnn.input_size) #T(B * seq_len, C, H, W)
    inputs = model.cnn.extract_feature(inputs) #T(B * seq_len, ft_size)

    outputs = model.process_feature(inputs) #T(Bn, A)
    loss = criterion( outputs, labels )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        _, pred = torch.max( outputs, dim = -1 ) #T(Bn), T(Bn)
        correct = torch.sum( pred  == labels ).item() # The number of correct predictions

    return loss.item(), correct


def train_step_lrcn(model, batch_data, optimizer, criterion):
    '''
    Train 'lrcn' model for one step
    
    Arguments:
    - model: pytorch module object
    - batch_data: python dictionary of {'data' : video clip tensor, 'label' : video label tensor}
    - optimizer: pytorch optimizer object
    - criterion: pytorch loss function object 

    Returns:
    - loss: float, loss value
    - correct: int, the number of correctly predicted videos
    '''    
    model.train()
    
    inputs = batch_data['data'] # T(Bn, seq_len, *)
    labels = batch_data['label'] # T(Bn, 1)

    #Spatial feature extraction
    inputs = inputs.view(-1, *model.cnn.input_size) #T(B * seq_len, C, H, W)
    inputs = model.cnn.extract_feature(inputs) #T(B * seq_len, 512)
    inputs = inputs.view(-1, model.seq_len, model.ft_size) #T(B, seq_len, 512)

    outputs = model.forward_lstm(inputs) #T(B, seq_len, A) 

    outputs = outputs.view( -1, model.action_size ) #T(Bn * seq_len, A)
    expanded_labels = torch.cat( [torch.cat( [labels[i]] * model.seq_len ) for i in range(labels.size(0))] ) #T(Bn * seq_len)
    
    loss = criterion( outputs, expanded_labels )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        outputs = outputs.view( -1, model.seq_len, model.action_size ) #T(Bn * seq_len, A)
        _, pred = torch.max( torch.mean( F.softmax( outputs, dim = 2 ), dim = 1), dim = 1) #T(Bn), T(Bn)
        correct = torch.sum( pred  == labels.squeeze() ).item() # The number of correct predictions

    return loss.item(), correct


def train_model(model, optimizer, criterion, dataloader, timer, num_epoch = 1):
    '''
    Train 'lrcn' model for one step
    
    Arguments:
    - model: pytorch module object
    - batch_data: python dictionary of {'data' : video clip tensor, 'label' : video label tensor}
    - optimizer: pytorch optimizer object
    - criterion: pytorch loss function object 

    Returns:
    - loss: float, loss value
    - correct: int, the number of correctly predicted videos
    '''    
    prints_every = 20

    total_training_videos = len(dataloader.dataset)

    if args.model == "lrcn":
        train_step = train_step_lrcn
    elif args.model == "1dcnn":
        train_step = train_step_1dcnn

    for epoch in range(num_epoch):
        processed_videos = 0 # Count total processed video clips
        processed_videos_btw = 0 # Count processed video clips in between intermediate reports
        total_loss = 0 # Accumulate loss values
        total_correct = 0 # Accumulate the number of correctly predicted videos
        interval_correct = 0 
        interval_loss = 0

        for loop_count, (data, label) in enumerate(dataloader):
            # data : video clip tensor, T(B, seq_len, 3, 224, 224)
            # label : label tensor, T(B)
            batch_data = {'data':data.to(device), 'label':label.to(device)}
            cur_loss, correct = train_step(model, batch_data, optimizer, criterion)

            total_loss += cur_loss
            total_correct += correct
            interval_loss += cur_loss
            interval_correct += correct

            processed_videos_btw += data.size(0) # Add data batch size
            processed_videos += data.size(0) # Add data batch size

            if (loop_count + 1) % prints_every == 0:
                avr_loss = interval_loss / prints_every
                print("L: %.4f A: %.4f %d epoch (%d/%d) %s" % (avr_loss, interval_correct / processed_videos_btw, timer.epoch, processed_videos, total_training_videos, timer.elapsed()))
                interval_correct = 0
                interval_loss = 0
                processed_videos_btw = 0
        print('----------------------------------')
        logger.info('Epoch: %d Training Loss %s Accuracy %.3f' % (timer.epoch + 1, interval_loss / (loop_count+1), total_correct / total_training_videos))
        timer.epoch_count()

    
def save_test_info(text, file_name):
    # Output text to the file

    mode = 'a' if os.path.exists(file_name) else 'w'
    with open(file_name, mode) as f:
        f.write(text + '\n')

def save_val_result(timer, loss, accuracy, file_name):
    # Output metrics to the file in a formatted way

    mode = 'a' if os.path.exists(file_name) else 'w'
    with open(file_name, mode) as f:
        f.write('[%s] %d epoch - (A: %.3f, L: %.3f)\n' % (timer.elapsed(), timer.epoch, accuracy, loss))

def save_model_with_arguments(model, arg_dict, path):
    # Save model parameters and program arguments
    # Note that to load the model parameter weights, it should uncover one more step
    # e.g.)
    # dict = torch.load(path)
    # model.load_state_dict( dict['model'] ) # (Not) model.load_state_dict( dict )
    save_dict = {}
    if isinstance( model, MyDataParallel ):
        save_dict['model'] = model.module.state_dict()
    else:
        save_dict['model'] = model.state_dict()

    save_dict['model_arg'] = arg_dict 
    torch.save(save_dict, path)
    

def main():
    seq_len = args.seq # An unit video clip length (16 is used in the paper)
    action_size = 101 # Number of target action categories (101 for UCF101)
    stride = 8 # Video clip stride (e.g. If stride = 8, then VideoClip1: Frame[0,...,seq_len-1], VideoClip2: Frame[stride,...,seq_len-1+stride],..., VideoClip_n: Frame[n*stride,...,seq_len-1+n*stride]
    v_batch_size = args.b # Video clip batchsize
    cnn_model_name = "resnet18" # CNN backbone network

    model_save_dir = expr_dir
    ft_dir = os.path.join(expr_dir, 'temp')
    os.mkdir(ft_dir)

    val_result_file = os.path.join(model_save_dir, 'validation_result.txt') 

    video_dir = args.data

    video_list = {'val':'ucf101_split1_testVideos.txt',
                  'train' :'ucf101_split1_trainVideos.txt'}
  
    norm_mean = [0.485, 0.456, 0.406] # Imagenet mean
    norm_std = [0.229, 0.224, 0.225] # Imagenet std

    # Frame transformation specification for training data
    train_image_transforms = {
        'rescale': 256, # Firstly rescale frames
        'outsize': (224, 224), # Frames are cropped into (224, 224) size
        'crop': True, # Perform random crop
        'flip': False, # Horizontal flippling
        'mean': norm_mean, # Mean value for normalization
        'std': norm_std # Std value for normalization
    }

    # Frame transformation pseicification for validation data
    val_image_transforms = transforms.Compose([
        transforms.Resize(256), # Rescale frames
        transforms.CenterCrop(224), # Center crop into (224, 224) size
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std) # Normalize
    ])

    ### Train Data Loader
    train_dataset = TrainVideoDataSet(video_dir, video_list['train'], transform_param = train_image_transforms, seq_len = seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size = v_batch_size, shuffle=True, num_workers = 8, pin_memory = True)

    print("Train New Model")
    hidden_size = args.dim

    ### Create model
    if args.model == 'lrcn':
        model = LRCN(hidden_size = hidden_size, action_size = action_size, seq_len = seq_len, cnn_model=cnn_model_name, lstm_dropout = 0.0, cnn_dropout = 0.0)
    elif args.model == '1dcnn':
        model = ODCNN(hidden_size = hidden_size, action_size = action_size, seq_len = seq_len, cnn_model = cnn_model_name, dropout2 = 0.0, dropout1 = 0.0, t_kernel_size = args.tsize)

    if args.weight_file != None:
        model_weight = torch.load(args.weight_file, map_location='cpu')
        del model_weight['model_arg']
        model.load_state_dict( model_weight )
        print("Model weight is loaded from ", args.weight_file)

    logger.info("Experiment model :" + args.model)
    logger.info("program parameters:"+ str(args.__dict__))
    model_arg = {'hidden_size' : hidden_size, 'action_size' : action_size, 'seq_len' : seq_len, 'cnn_model' : cnn_model_name}
            

    ### Data Parallel
    if torch.cuda.device_count() > 1:
        logger.info("Operating in Data Parallel Mode on %d Gpus" % torch.cuda.device_count())
        model = MyDataParallel(model)
    else:
        logger.info("Operating on a Single GPU")

    model.to(device)


    label_smoothing = args.a > 0.0

    ### Loss function
    if label_smoothing:
        # Lable smoothing Cross Entropy Loss
        criterion = SmoothCrossEntropyLoss(alpha = args.a, num_cls = action_size)
    else:
        # Vanila Cross Entropy Loss
        criterion = nn.CrossEntropyLoss()

    ### Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    learning_rate = args.lr

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum = 0.9, nesterov=True, weight_decay = 0.00001 )
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=learning_rate)

    best_acc = 0
    timer = TimeChecker()
    timer.start()

    ### Output hyper parameters to the log file
    save_test_info(','.join( k + ':' + str(model_arg[k]) for k in sorted(model_arg.keys()) )\
                    + ' batch ' + str(args.b)\
                    + ' label smoothing %s' %('False' if not label_smoothing else ('True ' + str(args.a)))\
                    + ' optimizer ' + args.optim + ' lr ' + str(learning_rate)
                    + ' seq_len ' + str(args.seq)
                    + args.model
                    , val_result_file)

    logger.info("===============================Start=========================================")

    ### Output hyper parameters to the log file
    logger.info( '\n'.join( [ k + ':' + str(args.__dict__[k]) for k in sorted(args.__dict__) ]  ) )


    iter_count = 0

    while True:
        timer.start('inter_train')
        train_model(model, optimizer, criterion, train_dataloader, timer = timer, num_epoch=args.eval_interval) # Train model for 'eval_interval' epochs
        iter_count += 1
        logger.info("Total %s elapsed for %d epochs train And %s elapsed for %d epochs train" % (timer.elapsed(), iter_count * args.eval_interval, timer.elapsed('inter_train'), args.eval_interval))

        if iter_count >= args.eval_since:
            ac, ls = evaluate_model(logger, args.model, model, criterion, stride, ft_dir, video_dir, val_image_transforms, video_list['val'], device = device) # Evaluate model

            save_val_result(timer, ls, ac, val_result_file) # Output metric to the log file

            # Save the best model
            if ac > best_acc:
                best_acc = ac
                save_model_with_arguments(model, model_arg, os.path.join(model_save_dir, 'S%d_B%d_%s_iter%d_%.2f.pth' % (seq_len, v_batch_size, args.model, iter_count, best_acc)) )
                #torch.save(model, os.path.join(model_save_dir, 'best_param.pth'))

        if iter_count == args.iter:
            break

if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
