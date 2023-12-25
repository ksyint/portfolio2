
import torch


def topk_accuracy(output: torch.Tensor,target: torch.Tensor ,topk: tuple=(1,)) -> list:

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        print(output.shape, maxk)
        _,pred = output.topk(maxk,1,True,True)
        pred = pred.t()
        correct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())

        return res