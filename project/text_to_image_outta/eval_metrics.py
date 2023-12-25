import numpy as np
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

from scipy.stats import entropy

def calculate_activation_statistics(images,model,batch_size=128, dims=2048, # 수정해야하는 hyperparameter
                    cuda=False):
  model.eval()
  act=np.empty((len(images), dims))

  if cuda:
      batch=images.cuda()
  else:
      batch=images
  pred = model(batch)[0]

      # If model output is not scalar, apply global spatial average pooling.
      # This happens if you choose a diSmensionality not equal 2048.
  if pred.size(2) != 1 or pred.size(3) != 1:
      pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

  act= pred.cpu().data.numpy().reshape(pred.size(0), -1)

  mu = np.mean(act, axis=0)
  sigma = np.cov(act, rowvar=False)
  return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape,\
      'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape,\
      'Training and test covariances have different dimensions'

  # calculate sum squared difference between means
  diff = mu1 - mu2

  # calculate sqrt of product between cov
  covmean = linalg.sqrtm(sigma1.dot(sigma2))

  # check and correct imaginary numbers from sqrt
  if np.iscomplexobj(covmean):
      covmean = covmean.real

  # calculate score
  fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)

  return fid
 

def calculate_fid(images_real,images_fake,model):

  # calculate mean and covariance statistics
  mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
  mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)

  """get fretched distance"""
  fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
  return fid_value


# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-10):

  # calculate p(y)
  p_y = expand_dims(p_yx.mean(axis=0), 0)

  # kl divergence for each image
  kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
  # sum over classes
  sum_kl_d = kl_d.sum(axis=1)
  # average over images
  avg_kl_d = mean(sum_kl_d)
  # undo the logs
  is_score = exp(avg_kl_d)
  return is_score


# list of images
# numpy array with values ranging from 0 to 255.
def calculate_is(imgs, cuda=False, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)