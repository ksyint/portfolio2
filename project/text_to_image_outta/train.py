import torch
import matplotlib.pyplot as plt
import torchvision
import os
import numpy as np
from clip2.clip import clip
import torchvision.transforms as T
import time
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import BCELoss
from read_dataset import ZipDataset
from dataloader import get_dataloader
from network import Generator, Discriminator, weight_init
from train_utils import *
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn as nn
import torch.functional as F
from torchvision.models import inception_v3 as themodel
import numpy as np
import lpips
d_losses = []
g_losses = []

cos_sim = torch.nn.CosineSimilarity(dim=0)


def contrastive_loss_G(fake_image, clip_model, txt_embedding, device, tau=0.5):
    
    ################# Problem 4-(c). #################
    '''
    TODO: 
        (1) Calculate clip image embedding using clip_model and normed_img. You must know how to use 'clip' Library. 
        (2) Normalize image embedding (Hint: use some function in train_utils.py)
            and save to image_features
        (3) Implement L_ConG equation and save to L_cont. Note that h' in equation is txt_embedding
    '''

    denorm_fake_image = denormalize_image(fake_image)
    reshaped_img = clip_transform(224)(denorm_fake_image)
    reshaped_img = custom_reshape(reshaped_img)
    normed_img = clip_preprocess()(reshaped_img).to(device)
    with torch.no_grad():
        image_embedding = clip_model.encode_image(normed_img)
    image_features = normalize(image_embedding)

    cosine_sim = cos_sim(image_features, txt_embedding.unsqueeze(0))

    L_cont = torch.relu(cosine_sim - tau).mean()

    ################# Problem 4-(c). #################
    return L_cont




def contrastive_loss_D(g_out_align, txt_embedding, tau=0.5):
    ################# Problem 4-(d). #################
    '''
    TODO: Normalize embedding extracted from align_disciminator 
         (Hint: use 'normalize' function in train_utils.py) and save to model_features
        Note that f_s(x_j) in equation is g_out_align and h' is txt_embedding
    '''

    model_features = normalize(g_out_align)  # Normalizing the model features

    cosine_sim = cos_sim(model_features, txt_embedding.unsqueeze(0))

    L_cont = torch.relu(cosine_sim - tau).mean()

    ################# Problem 4-(d). #################
    return L_cont





def D_loss(real_image, fake_image, model_D, loss_fn, 
               use_uncond_loss, use_contrastive_loss, 
               gamma,
               mu, txt_feature,
               d_fake_label, d_real_label):
    
    loss_d_comp = {}

    
    ################# Problem 4-(b). #################
    '''
    TODO: 
        (1) Calculate unconditional loss with fake images and save to loss_g_comp['d_loss_fake_uncond']
        (2) Calculate unconditional loss with real images and save to loss_g_comp['d_loss_real_uncond']
        (3) Calculate conditional loss with fake images and save to loss_g_comp['d_loss_fake_cond']
        (4) Calculate conditional loss with real images and save to loss_g_comp['d_loss_real_cond']
        (5) With (3) and (4), calculate align_out from align discriminator to calculate contrastive loss
        Use loss_fn to calculate loss
    '''

    if use_uncond_loss:
        fake_logits, _ = model_D(fake_image)
        fake_logits = fake_logits.squeeze()
        d_loss_fake_uncond = loss_fn(fake_logits, d_fake_label) #


        loss_d_comp['d_loss_fake_uncond'] = d_loss_fake_uncond
        
        real_logits, _ = model_D(real_image)
        real_logits = real_logits.squeeze()
        d_loss_real_uncond = loss_fn(real_logits, d_real_label)

        loss_d_comp['d_loss_real_uncond'] = d_loss_real_uncond
        


    
    real_logits, real_logits_align = model_D(real_image, condition = mu)
    real_logits = real_logits.squeeze()
    real_logits_align = real_logits_align.squeeze()
    #real_logits.requires_grad_ = True
    
    d_loss_real_cond = loss_fn(real_logits, d_real_label) 
    loss_d_comp['d_loss_fake_cond'] = d_loss_real_cond

     
    fake_logits, fake_logits_align = model_D(fake_image, condition = mu)
    fake_logits = fake_logits.squeeze()
    fake_logits_align = fake_logits_align.squeeze()
    #fake_logits.requires_grad_ = True
    
    d_loss_fake_cond = loss_fn(fake_logits, d_fake_label) 
    loss_d_comp['d_loss_real_cond'] = d_loss_fake_cond


    
    if use_contrastive_loss:
        d_loss_fake_cond_contrastive = gamma * contrastive_loss_D(fake_logits_align, mu)
        loss_d_comp['d_loss_fake_cond_contrastive'] = d_loss_fake_cond_contrastive
        
        d_loss_real_cond_contrastive = gamma * contrastive_loss_D(real_logits_align, mu)
        loss_d_comp['d_loss_real_cond_contrastive'] = d_loss_real_cond_contrastive

    ################# Problem 4-(b). #################
    d_loss = gather_all(loss_d_comp)
    return d_loss







def G_loss(fake_image, model_D, loss_fn,
           use_uncond_loss, use_contrastive_loss,
           clip_model, gamma, lam, device,
           mu, txt_feature,
           g_label):
    
    loss_g_comp = {}

    
    ################# Problem 4-(a). #################
    '''
    TODO: 
        (1) Calculate unconditional loss and save to loss_g_comp['g_loss_uncond']
        (2) Calculate conditional loss and save to loss_g_comp['g_loss_cond']  
        (3) With (2), calculate align_out from align discriminator to calculate contrastive loss
        Use loss_fn to calculate loss
    '''
    if use_uncond_loss:
        fake_logits, _ = model_D(fake_image)
        fake_logits = fake_logits.squeeze()
        g_loss_uncond = loss_fn(fake_logits, g_label) 
        
        loss_g_comp['g_loss_uncond'] = g_loss_uncond
    
    
    fake_logits, fake_logits_align = model_D(fake_image, condition = mu) 
    fake_logits = fake_logits.squeeze() 
    fake_logits_align = fake_logits_align.squeeze()
    #real_logits.requires_grad = True
    
    g_loss_cond = loss_fn(fake_logits, g_label)
    loss_g_comp['g_loss_cond'] = g_loss_cond
    
    # real_logits=torch.tensor(real_logits[0]).squeeze()
    # real_logits.requires_grad = True


    if use_contrastive_loss:
        if fake_image.shape[-1] >= 256: 
            g_loss_cond_contrastive = lam * contrastive_loss_G(fake_image, clip_model, txt_feature, device)
            loss_g_comp['g_loss_cond_contrastive'] = g_loss_cond_contrastive
            
        d_loss_cond_contrastive = gamma * contrastive_loss_D(fake_logits_align, txt_feature)    
        loss_g_comp['d_loss_cond_contrastive'] = d_loss_cond_contrastive
    
    ################# Problem 4-(a). #################

    g_loss = gather_all(loss_g_comp)
    return g_loss





def train_step(train_loader, noise_dim, device, model_G, model_D_lst, optim_g, optim_d_lst, 
               loss_fn, num_stage, use_uncond_loss, use_contrastive_loss, report_interval, 
               clip_model, gamma, lam):
    
    d_loss_train = 0
    g_loss_train = 0

    for iter, batch in enumerate(train_loader):
        real_imgs, img_feature, txt_feature = batch
        if iter == 0: save_txt_feature = txt_feature

        BATCH_SIZE = real_imgs[-1].shape[0]
        for i in range(num_stage): real_imgs[i] = real_imgs[i].to(device)

        img_feature = img_feature.to(device)
        txt_feature = txt_feature.to(device)


        
        ################# [Optional]] Problem 4-(f). #################
        '''
        TODO: pseudo text feature generation for Language-free training
        Generate the pseudo text feature using the idea of 'fixed perturbations' of LAFITE (https://arxiv.org/pdf/2111.13792.pdf).
        Note that img_feature and txt_feature is already normalized.
        '''

        perturbations = torch.randn_like(txt_feature) * 0.1  # You can adjust the perturbation magnitude
        pseudo_txt_feature = txt_feature + perturbations

        # Normalize the pseudo text feature
        pseudo_text_feature = normalize(pseudo_txt_feature)
        ################# Problem 4-(f). #################

        

        ################# Problem 4-(e). #################
        '''
        TODO: Generate label for loss calculation
        (1) Use torch.zeros or torch.ones
        (2) Cast dtype into torch.float32
        (3) Move the tensor into device
        '''

        d_fake_label = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float32)
        d_real_label = torch.ones(BATCH_SIZE, device=device, dtype=torch.float32)
        g_label = torch.ones(BATCH_SIZE, device=device, dtype=torch.float32)
        ################# Problem 4-(e). #################





        # Phase 1. Optmize Discriminator
        noise = torch.randn(BATCH_SIZE, noise_dim).to(device)
        fake_images, mu, log_sigma = model_G(txt_feature, noise)
        d_loss = 0

        for i in range(num_stage):
            optim_d = optim_d_lst[i]
            optim_d.zero_grad()

            d_loss_i = D_loss(real_imgs[i], fake_images[i], model_D_lst[i], loss_fn, 
                              use_uncond_loss, use_contrastive_loss,
                              gamma,
                              mu, txt_feature,
                              d_fake_label, d_real_label)
            d_loss += d_loss_i.detach().item()
            d_loss_i.backward(retain_graph=True)
            optim_d.step()
            d_loss_train += d_loss_i.item()




        # Phase 2. Optimize Generator
        optim_g.zero_grad()
        noise = torch.randn(BATCH_SIZE, noise_dim).to(device)
        fake_images, mu, log_sigma = model_G(txt_feature, noise)
        g_loss = 0

        for i in range(num_stage):
            g_loss_i = G_loss(fake_images[i], model_D_lst[i], loss_fn,
                              use_uncond_loss, use_contrastive_loss,
                              clip_model, gamma, lam, device,
                              mu, txt_feature,
                              g_label)
            g_loss += (0.1)*g_loss_i #1.0
        

        # Calculation of L_CA. Do NOT modify.
        aug_loss = KL_divergence(mu, log_sigma)
        g_loss += (1.0) * aug_loss #1.0
        
#         criterion=nn.L1Loss().cuda()
#         for i in range(num_stage):
            
#             g_lossi=criterion(fake_images[i],real_imgs[i])
#             g_loss+=g_lossi
        criterion=nn.L1Loss().cuda()
        criterion0=nn.MSELoss().cuda()
        l1loss=criterion(fake_images[0],real_imgs[0])
        
        
        
        
        
        g_loss+=(1.0)*l1loss
       
        
        
        g_loss.backward()
        optim_g.step()
        g_loss_train += g_loss.item()


        
        # loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
        # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

        # import torch
        # img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
        # img1 = torch.zeros(1,3,64,64)
        # d = loss_fn_alex(img0, img1)


        # Phase 3. Report
        if iter % report_interval == 0 and iter >= report_interval:
            print(f"    Iteration {iter} \t d_loss: {(d_loss):.4f}, g_loss: {(g_loss.item()):.4f}")
    


    d_loss_train /= len(train_loader)
    g_loss_train /= len(train_loader)
    d_losses.append(d_loss_train)
    g_losses.append(g_loss_train)

    return d_loss_train, g_loss_train, save_txt_feature


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clip_embedding_dim = args.clip_embedding_dim # Dimension of c_txt, default: 512 (CLIP ViT-B/32)
    projection_dim = args.projection_dim # Dimension of \hat{c_txt} extracted from CANet, default: 128
    noise_dim = args.noise_dim # Dimension of noise z ~ N(0, 1), default: 100 
    g_in_chans = 1024 # Equal to Ng
    g_out_chans = 3 # Fixed
    d_in_chans = 64 # Equal to Nd
    d_out_chans = 1 # Fixed
    num_stage = args.num_stage # default: 3
    use_uncond_loss = args.use_uncond_loss
    use_contrastive_loss = args.use_contrastive_loss
    report_interval = args.report_interval # default: 100
    incept = InceptionScore()
    fidmetirc = FrechetInceptionDistance(feature=192)

    save_hyp(args, g_in_chans, g_out_chans, d_in_chans, d_out_chans)

    print("Loading dataset")
    train_dataset = ZipDataset(args.train_data, num_stage)
    train_loader = get_dataloader(args=args, dataset=train_dataset, is_train=True)
    print("finish")


    G = Generator(clip_embedding_dim, projection_dim, noise_dim, g_in_chans, g_out_chans, num_stage, device).to(device)
    G.apply(weight_init)

    D_lst = [
        Discriminator(projection_dim, g_out_chans, d_in_chans, d_out_chans, clip_embedding_dim, curr_stage, device).to(device)
        for curr_stage in range(num_stage)
    ]
    for D in D_lst:
        D.apply(weight_init)
    
    if args.resume_checkpoint_path is not None and args.resume_epoch != -1:
        load_checkpoint(G, D_lst, args.resume_checkpoint_path, args.resume_epoch)
        print('Resumed from saved checkpoint')

    lr = args.learning_rate
    num_epochs = args.num_epochs

    # NOTE: You may try different optimizer setting or use learning rate schduler
    optim_g = Adam(G.parameters(), lr = lr, betas = (0.5, 0.999))
    optim_d_lst = [
        Adam(D_lst[curr_stage].parameters(), lr = lr, betas = (0.5, 0.999))
        for curr_stage in range(num_stage)
    ]
    loss_fn = BCELoss()

    clip_model, _ = clip.load("ViT-B/32", device=device)

    for epoch in range(args.resume_epoch + 1, num_epochs):
        print(f"Epoch: {epoch} start")
        start_time = time.time()
        d_loss, g_loss, txt_feature = train_step(train_loader, noise_dim, device, G, D_lst, optim_g, optim_d_lst, 
                                                 loss_fn, num_stage, use_uncond_loss, use_contrastive_loss, report_interval,
                                                 clip_model, gamma=5, lam=10)
        end_time = time.time()
        print(f"Epoch: {epoch} \t d_loss: {d_loss:.4f} \t g_loss: {g_loss:.4f} \t esti. time: {(end_time - start_time):.2f}s")

        # sampling : generate fake images and save
        print("evaluation")
        with torch.no_grad():
            
            texts = []
            for batch in train_loader:
                reali, imgf, txtf = batch
                texts.append(txtf)
            thelist=[]
            thelist2=[]    
            length=len(texts)
            for i in range(length):
                txt_feature=texts[i]
                z = torch.randn(txt_feature.shape[0], noise_dim).to(device)
                txt_feature = txt_feature.to(device)

                fake_images, _, _ = G(txt_feature, z)
                
#                 # IS score
#                 T=nn.Upsample(size=(64,64))
#                 T2=nn.Upsample(size=(299,299))
#                 #nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
#                 T3=nn.Upsample()

                
#                 for i in range(len(fake_images)):
    
#                     fake_images2=torch.tensor(fake_images[i],dtype=torch.uint8)
#                     fake_images_for_IS = T2(fake_images2.detach().cpu())
#                     incept.update(fake_images_for_IS)
#                     IS_score = incept.compute()
#                     thelist.append(IS_score[0])
#                     thelist2.append(IS_score[1])


            
#             # FID score
#             ris = []
#             for batch in train_loader:
#                 reali, imgf, txtf = batch
#                 ris.append(reali)
#             length2=len(ris)
#             thelist3=[]
#             for i in range(length2):
#                 #print(len(ris[i]))
#                 real_images = ris[i][np.random.randint(0,1)]
#                 real_images=torch.tensor(real_images,dtype=torch.uint8)
#                 fidmetirc.update(real_images, real= True)
#                 fidmetirc.update(fake_images_for_IS, real = False)
#                 FID_score = fidmetirc.compute()
#                 thelist3.append(FID_score)

#             print(f"Epoch: {epoch} \t IS_mean: {np.mean(thelist):.4f} \t IS_std: {np.mean(thelist2)} \t FID_score: {np.mean(thelist3):.4f} ")

            fake_image = fake_images[-1].detach().cpu() # visulize only the high-res images
            epoch_ret = torchvision.utils.make_grid(fake_image, padding=2, normalize=True)
            torchvision.utils.save_image(epoch_ret, os.path.join(args.result_path, f"epoch_{epoch}.png"))

        # save checkpoint
        save_model(args, G, D_lst, epoch, num_stage)
