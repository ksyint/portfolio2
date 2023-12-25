import torch.utils.data as data
import numpy as np
from utils.utils import process_feat
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import option
import glob
args=option.parse_args()

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, is_preprocessed=False):
        self.modality = args.modality
        self.is_normal = is_normal
        
        if test_mode:
            self.rgb_list_file = sorted(glob.glob("./warn/*.npy"))
        else:
            if is_normal==True:
                self.rgb_list_file = glob.glob("./datas/normal/*.npy")
            
            else: #abnormal but warning
                self.rgb_list_file = glob.glob("./warn2/*.npy")+glob.glob("./datas/abnormal/*.npy")
                
        
        self.tranform = transform
        self.test_mode = test_mode
        self.list=self.rgb_list_file
        self.num_frame = 0
        self.labels = None
        self.is_preprocessed = args.preprocessed

    


    def __getitem__(self, index):
        label = self.get_label(index)  # get video level label 0/1
        if args.datasetname == 'UCF':
            features = np.load(self.list[index], allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1]
            name=name.split(".")[-2]
        
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            if args.datasetname == 'UCF':
                mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
                features = np.concatenate((features,mag),axis = 2)
            
            return features, name
        else:
            if args.datasetname == 'UCF':
                if self.is_preprocessed:
                    return features, label
                if features.ndim==4:
                    features=features.squeeze(3)
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []

                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, args.seg_length) #ucf(32,2048)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate((divided_features,divided_mag),axis = 2)
                return divided_features, label


    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def __len__(self):

        return len(self.list)


    def get_num_frames(self):
        return self.num_frame
