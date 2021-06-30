from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch

import random
import os
from PIL import Image
import numpy as np
import h5py

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class VideoData(VisionDataset):
    """
    VideoData class

    It reads every frame of every video clip under the 'root' directory
    
    Arguments:
    - root (str): the root directory for video clips
    - video_list (str): the file name which contains video clip names and their corresponding action labels
    - transform (callable): pytorch transform object to be applied to each frame
                            transform should not have randomness for consistent transform operations to frames in a video clip
    """
    def __init__(self, root, video_list, transform=None, loader=default_loader):
        super(VideoData, self).__init__(root, transform=transform)


        def parsing_line( line ):
            p, l = line.strip().split(' ')
            c , vname = p.split('/')
            return c, vname, int(l)

        ### Read the file, video_list
        with open(video_list, 'r') as f:
            self.videos = {vname: label for _, vname, label in map(parsing_line, f)}
        
        #self.vnames = [ x for x in self.videos.keys() ]

        samples = []
        ### Retrieve video clip subfolders under the 'root' directory
        for vname in os.listdir( root ):
            if vname in self.videos:
                vpath = os.path.join(root, vname)
                label = self.videos[vname]
                start_idx = len( samples )
                for frame in os.listdir( vpath ):
                    path = os.path.join(vpath, frame)
                    samples.append( (path, label) )
                end_idx = len( samples )
                self.videos[vname] = (label, start_idx, end_idx)


        self.samples = samples
        self.transform = transform
        self.loader = loader


    def video_unit_batch_sampler(self):
        '''
        Every iteration, this sampler returns a generator for indicies from the starting frame index to the last frame index for each video clip
        '''
        return iter([range(s, e) for _, s, e in map(lambda v: self.videos[v], self.videos.keys())])


#    def num_frame(v_index):
#        _, indices = self.videos[ self.vnames[v_index] ]
#        return len(indices)


    def video_names(self):
        return iter( self.videos.keys() )
#    '''
#    def get_video_name(v_index):
#        return self.vnames[v_index]
#    '''
    def __getitem__(self, index):
        '''
        Load the frame corresponding to the index and return with its label
        '''
        path, target = self.samples[index]
        sample = self.loader(path) # Load the frame
        if self.transform is not None:
            sample = self.transform(sample) # Apply transform to the frame

        return sample, target # Return the frame sample and its label

    def __len__(self):
        return len(self.samples)


def load_frame(frame_dir, video_name, index):
    fname = "%s.%04d.jpg" % (video_name, index)
    with open( os.path.join(frame_dir, fname), 'rb') as imf:
        return Image.open( imf ).convert('RGB')


def transform_frames( frames, rescale_size, output_size, crop, flip, mean, std ):
    '''
    Apply consistent transforms to frames in a video clip

    Arguments:
    - frames (list): frame list
    - rescale_size: parameter for a resize transform
    - output_size: the target size of frames
    - crop: If it is true, random cropping is applied
    - flip: If it is true, horizontal flipping is applied with a chance of 0.5
    - mean: mean value for the normalization
    - std: std value for the normalization
    '''
    normalizer = transforms.Normalize(mean, std)
    resizer = transforms.Resize( rescale_size )
    flip = flip and (random.random() > 0.5)

    i = 1
    t = []
    for f in frames:
        f = resizer(f)
        if crop:
            random_crop_param = transforms.RandomCrop.get_params(f, output_size)
            f = TF.crop(f, *random_crop_param)
        if flip:
            f = TF.hflip(f)
        f = TF.to_tensor(f)
        f = normalizer( f )
        t.append( f )
        i += 1

    return torch.stack(t)





class TrainVideoDataSet(Dataset):
    """
    Arguments:
    - video_dir (str): the root directory of video clips
    - video_list (str): the file name for a training video clip list
    - seq_len (int): Unit video clip length
    - transform_param (dict)
    """
    def __init__(self, video_dir, video_list, seq_len, transform_param):
        self.vdir = video_dir
        self.labels = {}
        self.vnames = []
        self.vpaths = []
        with open(video_list, 'r') as f:
            for line in f:
                line = line.strip()
                vname = line.split(' ')[0].split('/')[-1]
                label = int( line.split(' ')[1] )
                path = os.path.join(video_dir, vname)
                if vname not in self.labels:
                    if not (os.path.exists( path ) or os.path.isdir( path )):
                        raise Exception("Invalide Video path %s" % path)
                    self.labels[vname] = label
                    self.vnames.append(vname)
                    self.vpaths.append( path )
                elif label != self.labels[vname]:
                    raise Exception("Invalid Video list - %s appears with different labels in the list" % vname)

        self.seq_len = seq_len
        
        self.rescale_size = transform_param.get('rescale', 256)
        self.output_size = transform_param.get('outsize', (224, 224))
        self.crop = transform_param.get('crop', True)
        self.flip = transform_param.get('flip', False)
        self.mean = transform_param.get('mean', [0.485, 0.456, 0.406])
        self.std = transform_param.get('std', [0.229, 0.224, 0.225])

    def __len__(self):       
        return len(self.vnames)

    def __getitem__(self, index):
        """
        Return consecutive frames of the video clip corresponding to 'index' with the label
        Frames are choosen at random position
        Frames have the length, seq_len
        """
        path = self.vpaths[index] 
        vname = self.vnames[index]
        num_frames = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and os.path.join(path,f).endswith('.jpg')])
        if num_frames < self.seq_len:
            raise Exception("At least %d frames are required to form a clip" % self.seq_len)
        
        frames = []
        start_frame = random.randint(1, num_frames - self.seq_len + 1) # Choose starting position randomly

        for frame_index in range(start_frame, start_frame + self.seq_len):
            frames.append( load_frame(path, vname, frame_index) ) # Load consecutive frames

        frames = transform_frames( frames, self.rescale_size, self.output_size, self.crop, self.flip, self.mean, self.std ) # Apply transforms
        label = torch.LongTensor( [self.labels[self.vnames[index]]] )

        return frames, label # Return frames and the label


#def abs_from_generator(boundaries, stride, seq_len, boundaries_in_clips = None):
#    prev_bound = 0
#    clip_count = 0
#    for next_bound in boundaries:
#        for abs_start in range(prev_bound, next_bound - seq_len, stride):
#            clip_count += 1
#            yield abs_start
#        if next_bound - seq_len > abs_start:
#            clip_count += 1
#            yield next_bound - seq_len
#        if boundaries_in_clips is not None:
#            boundaries_in_clips.append( clip_count )
#        prev_bound = next_bound

class ClipFeatureDataSet():
    """
    ClipFeatureDataSet loads spatial features from the h5py file and returns

    Arguments:
    - ft_dir (str): the directory where feature files locate
    - video_list (str)
    - seq_len (int)
    - stride (int)
    - batch_size (int)
    """
    def __init__(self, ft_dir, video_list, seq_len, stride, batch_size):
        self.where_ft = {}


        for files in os.listdir(ft_dir):
            if files.endswith(".h5"):
                full_path = os.path.join(ft_dir, files)
                h5f = h5py.File(full_path)
                for vname in h5f.keys():
                    self.where_ft[vname] = full_path
                h5f.close()

        self.label = {}

        with open(video_list, 'r') as f:
            for line in f:
                line, label = line.split(' ')
                vname = line.split('/')[-1]
                self.label[vname] = int(label)

        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size


    def get_features(self, vname):
        h5fpath = self.where_ft[vname]

        h5f = h5py.File(h5fpath, 'r')
        t = torch.from_numpy( np.array( h5f[vname] ) )
        h5f.close()
        return t


    def clip_gen(self, ft):
        nframe = ft.size(0)
        clips = []
        _debug_indices = []
        for idx in range(0, nframe - self.seq_len, self.stride): # Each video clip moves by self.stride and has the length of self.seq_len
            clips.append( ft[idx:idx + self.seq_len] )
            _debug_indices.append( (idx, idx + self.seq_len - 1) )

        idx = nframe - self.seq_len
        clips.append( ft[idx:idx + self.seq_len] )
        _debug_indices.append( (idx, idx + self.seq_len) )

        return clips, _debug_indices


    def generator(self):
        """
        This generator returns video clips
        Each video clip has the length, self.seq_len
        It returns video clips by the stride, self.stride
        """
        for vname in sorted(self.label.keys()):
            label = self.label[vname]
            ft = self.get_features( vname )
            clips, _debug_indices = self.clip_gen(ft)
            nclips = len(clips)
            for i in range(0, nclips, self.batch_size):
                yield torch.stack(clips[i:min(nclips,i + self.batch_size)]), vname, label, _debug_indices[i:min(nclips, i + self.batch_size)]
