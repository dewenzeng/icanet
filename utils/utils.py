import time
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def profile_network_latency(model, input):
    target_device = input.device
    lt = []
    # print(f'target_device:{target_device}')
    with torch.no_grad():
        if model is None:
            raise NotImplementedError
        else:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for i in range(5000):
                if 'cuda' in str(target_device):
                    # use synchronize()
                    torch.cuda.synchronize()
                    # start_time = time.time()
                    starter.record()
                    _ = model(input)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                else:
                    start_time = time.time()
                    _ = model(input)
                    end_time = time.time()
                estimated_latency = curr_time
                lt.append(estimated_latency)
        lt.sort()
        lt = lt[1000:-1000]
        return sum(lt) / 3000

def dice_iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, N_class):
    """Compute the dice and iou with torch tensor. 
    Only support 3D data, please move all inputs to CPU beforehand.
    
    Args:
        outputs: network output after argmax, [N,Z,H,W].
        labels: ground truth, [N,Z,H,W].
        N_class: number of classes.

    Returns:
        Dice: mean dice of N sampels.
        IoU: mean iou of N samples.
    """
    SMOOTH = 1e-5
    outputs = outputs.float()
    labels = labels.float()
    dice = torch.ones(N_class-1).float()
    iou = torch.ones(N_class-1).float()
    ## for test
    #outputs = torch.tensor([[1,1],[3,3]]).float()
    #labels = torch.tensor([[0, 1], [2, 3]]).float()

    for iter in range(1,N_class): ## ignore the background
        predict_temp = torch.eq(outputs, iter)
        label_temp = torch.eq(labels, iter)
        intersection = predict_temp & label_temp
        intersection = intersection.float().sum((1,2,3))
        union_dice = (predict_temp.float().sum((1,2,3)) + label_temp.float().sum((1,2,3)))
        union_iou = (predict_temp | label_temp).float().sum((1,2,3))
        dice[iter-1] = ((2 * intersection + SMOOTH) / (union_dice + SMOOTH)).mean()
        iou[iter-1] = ((intersection + SMOOTH) / (union_iou + SMOOTH)).mean()
    return dice.mean(), iou.mean()  # Or thresholded.mean()
