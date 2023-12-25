import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from model import Model
from torchvision import transforms
from PIL import Image 
import numpy as np 
import glob 
from torch.utils.data import DataLoader


T1=transforms.ToTensor()

class The_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        
        self.list=glob.glob("/home/ksyint/other1213/1221/dataset/finetune/*/*.png")+glob.glob("/home/ksyint/other1213/1221/dataset/finetune/*/*.jpg")+glob.glob("/home/ksyint/other1213/1221/dataset/finetune/*/*.JPG")
        new_list=[]
        for img in self.list:
             for img2 in self.list:
                new_list.append(img+" "+img2)

        self.list2=new_list

       
    def __getitem__(self, idx):

        img1_path,img2_path=self.list2[idx].split(" ")

        img1=Image.open(img1_path)
        img1=np.array(img1)
        img1=img1[:,:,0:3]
        img1=Image.fromarray(img1)
        img1=img1.resize((224,224))
        img1=T1(img1)

        img2=Image.open(img2_path)
        img2=np.array(img2)
        img2=img2[:,:,0:3]
        img2=Image.fromarray(img2)
        img2=img2.resize((224,224))
        img2=T1(img2)

        img1_folder_name=img1_path.split("/")[-2]
        img2_folder_name=img2_path.split("/")[-2]

        if img1_folder_name!=img2_folder_name:
            label=torch.tensor([0])
        else:
            label=torch.tensor([1])
            
         
        return img1,img2,label


    def __len__(self):
        return len(self.list2)



import wandb




def main():
    wandb.init(project='davit huge')
    # 실행 이름 설정
    wandb.run.name = 'run2'
    wandb.run.save()
   


    device=torch.device("cuda:0")

    model=Model()
    #model=torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("davithugestate2.pth"))
    model=model.to(device)

    TrainDataset=The_Dataset()
    TrainLoader=DataLoader(TrainDataset,batch_size = 14,shuffle = True)
    

    optimizer=torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)
    best_number=10000
    torch.manual_seed(1)

    import time 
    start_time=time.time()
    for epoch in range(1000):
        model.train()
        number=0
        for index,data in enumerate(TrainLoader):
            img1,img2,label=data
          
            label=label.squeeze(1)
            img1=img1.to(device)
            img2=img2.to(device)
            label=label.to(device)
            
            loss=model(img1,img2,label)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            number+=loss.item()
            #wandb.log("Train batch loss")
            wandb.log({"Train batch loss": loss.item()})
            with open("result3.txt","a") as result1:
                result1.writelines(str(loss.item()))
                result1.writelines("\n")
                result1.close()
       
        train_loss=number/len(TrainLoader)
        print(f"Epoch{epoch+1}: {train_loss}")
        wandb.log({"Train loss": train_loss})
        with open("result_epoch3.txt","a") as result2:
                result2.writelines(str(train_loss))
                result2.writelines("\n")
                result2.close()

        current_time=time.time()
        wandb.log({"Wasted time": current_time-start_time})
        if train_loss<best_number:
            torch.save(model.state_dict(),"davithugestate3.pth")
            torch.save(model,"davithuge3.pth")
            best_number=train_loss


if __name__ == "__main__":
    main()
