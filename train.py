import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime
from models.RecursiveUNet3D import UNet3D
from utils.utils import *
from datasets.acdc import get_ACDC_dataset_generator
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='acdc_training', help="Setup experiment name")
parser.add_argument('--log_dir', type=str, default='runs', help="tensorboard summary dir")
parser.add_argument('--model_dir', type=str, default='checkpoints', help="model checkpoint dir")
parser.add_argument('--train-batch-size', default=4, type=int,
                help='training mini-batch size (default: 4)')
parser.add_argument('--val-batch-size', default=4, type=int,
                help='validation mini-batch size (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                metavar='W', help='weight decay (default: 1e-4)',
                dest='weight_decay')
parser.add_argument('--min_lr', default=0.0, type=float, 
                    help='minimal learning rate', dest='min_lr')
parser.add_argument('--num_classes', default=4, type=int,
                help='number of classes in the dataset.')

def main():
    
    args = parser.parse_args()

    # Setup random seeds.
    np.random.seed(1)
    cudnn.benchmark = True
    torch.manual_seed(1)
    cudnn.enabled = True
    torch.cuda.manual_seed(1)

    # check if gpu training is available
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')

    # dataset
    train_gen, val_gen = get_ACDC_dataset_generator(args.train_batch_size, args.val_batch_size)
    # hard code here, move them to args if you want to change them through argument.
    train_num_batches_per_epoch = 80
    val_num_batches_per_epoch = 20

    # Define model    
    model = UNet3D(num_classes=args.num_classes, in_channels=1, initial_filter_size=32, kernel_size=3, num_downs=4, norm_layer=nn.InstanceNorm3d)

    # Use model parallel?
    # model = nn.DataParallel(model)

    model = model.to(args.device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)
    # Define criterion
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr,
        last_epoch=-1)
    
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dir, args.experiment_name+'_'+current_time)
    writer = SummaryWriter(log_dir)

    # track the global iterations.
    args.n_iter = 0

    def train_one_epoch():
        dice_score = AverageMeter()
        train_gen.restart()
        model.train()
        for batch_idx in tqdm(range(train_num_batches_per_epoch)):
            data_dict = next(train_gen)
            images = data_dict['data']
            target = data_dict['target']
            if not isinstance(images, torch.Tensor):
                images = torch.from_numpy(images).float()
            if not isinstance(target, torch.Tensor):
                target = torch.from_numpy(target).float()
            images, target = images.cuda(), target.cuda()
            output = model(images)
            loss = criterion(output, target.long().squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get evaluation result
            output_softmax = F.softmax(output, dim=1)
            output_argmax = torch.argmax(output_softmax, dim=1).cpu()
            dice, _ = dice_iou_pytorch(output_argmax, target.cpu().squeeze(dim=1), args.num_classes)
            dice_score.update(dice.item(), images.shape[0])
            # writer loss to the log.
            writer.add_scalar('training_loss', loss, global_step=args.n_iter)
            args.n_iter += 1
        return dice_score.avg

    def val_one_epoch():
        dice_score = AverageMeter()
        model.eval()
        val_gen.restart()
        with torch.no_grad():
            for batch_idx in tqdm(range(val_num_batches_per_epoch)):
                data_dict = next(val_gen)
                images = data_dict['data']
                target = data_dict['target']
                keys = data_dict['keys']
                # print(f'keys:{keys}')
                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(images).float()
                if not isinstance(target, torch.Tensor):
                    target = torch.from_numpy(target).float()
                images, target = images.cuda(), target.cuda()
                output = model(images)
                # get evaluation result
                output_softmax = F.softmax(output, dim=1)
                output_argmax = torch.argmax(output_softmax, dim=1).cpu()
                dice, _ = dice_iou_pytorch(output_argmax, target.cpu().squeeze(dim=1), args.num_classes)
                dice_score.update(dice.item(), images.shape[0])
        return dice_score.avg

    best_dice = 0
    for epoch in range(args.epochs):
        train_dice = train_one_epoch()
        val_dice = val_one_epoch()
        print(f'Epoch {epoch}/{args.epochs} \t train_dice {train_dice:.3f} \t val_dice:{val_dice:.3f}')
        scheduler.step()
        writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
        writer.add_scalar('train_dice', train_dice, global_step=epoch)
        writer.add_scalar('val_dice', val_dice, global_step=epoch)
        # save the best checkpoint
        if best_dice < val_dice:
            best_dice = val_dice
            saved_result = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }
            ckpt_dir = os.path.join(args.model_dir, args.experiment_name+'_'+current_time)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(saved_result, os.path.join(ckpt_dir, 'best.pth'))

    print("Training has finished.")
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
