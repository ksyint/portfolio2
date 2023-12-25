from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm

from config import *
from models.mgfn import mgfn as Model
from datasets.dataset import Dataset

args = option.parse_args()
config = Config(args)

torch.multiprocessing.set_start_method('spawn')
device=torch.device("cuda")
test_loader = DataLoader(Dataset(args, test_mode=True),
                               batch_size=1, shuffle=False,
                               num_workers=0, pin_memory=False)
model=torch.load("warn4.pth")
with torch.no_grad():
    thelist=[]
    
    model.eval()
    pred = torch.zeros(0)
    featurelen =[]
    wrong=[]
    for i, inputs in tqdm(enumerate(test_loader)):
        if i<50:
            
            input = inputs[0].to(device)
            #print(input.shape)
            if input.ndim==5:
                input=input[:,:,:,:,0]
            input = input.permute(0, 2, 1, 3)
            logits = model(input)[-1]
            
            
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            print(torch.mean(sig).item())
            if torch.mean(sig).item()<0.15:
                wrong.append(i+1)
            thelist.append(torch.mean(sig).item())
            featurelen.append(len(sig))
            
            
            if type(pred)!=torch.Tensor:
                torch.from_numpy(pred)
            else:
                continue
            pred = torch.cat((pred, sig))


            pred = list(pred.cpu().detach().numpy())
            #print(np.mean(pred))
            #thelist.append(np.mean(pred))
            pred = np.repeat(np.array(pred), 16)
           
        elif i==50:
            break
    print("평균 값",np.mean(thelist))    
    print(wrong)
      

        
