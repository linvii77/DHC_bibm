import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
# Hybrid proxy arguments
parser.add_argument('--use_variation', action='store_true', default=False)
parser.add_argument('--lambda_cs', type=float, default=0.1)
parser.add_argument('--num_variations', type=int, default=5)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--patience', type=int, default=None)  # overrides config.early_stop_patience
parser.add_argument('--lambda_cs_rampup', type=int, default=0,
                    help='epochs to linearly ramp lambda_cs from 0 to target (0=no rampup)')
parser.add_argument('--variation_warmup', type=int, default=0,
                    help='epochs before variation_active is set to True (0=always active)')
parser.add_argument('--tau_var', type=float, default=10.0,
                    help='softmax temperature for variation sub-distribution (default 10.0; try 5.0 for softer assignment)')
parser.add_argument('--max_samples_per_class', type=int, default=0,
                    help='class-balanced sampling in CSL: max voxels per class (0=disabled)')
parser.add_argument('--pseudo_proxy', action='store_true', default=False,
                    help='use unlabeled pseudo-labels for proxy loss (backbone detached)')
parser.add_argument('--pseudo_proxy_warmup', type=int, default=200,
                    help='epoch to start pseudo_proxy (proxy centers must be stable first)')
parser.add_argument('--pseudo_proxy_w', type=float, default=0.05,
                    help='weight for pseudo_proxy loss term')
parser.add_argument('--pseudo_proxy_w_rampup', type=int, default=50,
                    help='epochs to linearly ramp pseudo_proxy_w from 0 to target after warmup')
parser.add_argument('--lambda_sac', type=float, default=0.0,
                    help='weight for SAC (Semantic Anchor Constraint) loss on proxy means (0=disabled)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.vnet import VNet
from proxy_loss import ProjectionHead, CompositionalSimilarityLoss
from utils import EMA, maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, print_func, kaiming_normal_init_weight
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config
config = Config(args.task)
if args.patience is not None:
    config.early_stop_patience = args.patience



def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w



def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size, args.task),
                RandomFlip_LR(),
                RandomFlip_UD(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size, args.task),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
    model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    return model, optimizer




class DistDW:
    def __init__(self, num_cls, do_bg=False, momentum=0.95):
        self.num_cls = num_cls
        self.do_bg = do_bg
        self.momentum = momentum

    def _cal_weights(self, num_each_class):
        num_each_class = torch.FloatTensor(num_each_class).cuda()
        P = (num_each_class.max()+1e-8) / (num_each_class+1e-8)
        P_log = torch.log(P)
        weight = P_log / P_log.max()
        return weight

    def init_weights(self, labeled_dataset):
        if labeled_dataset.unlabeled:
            raise ValueError
        num_each_class = np.zeros(self.num_cls)
        for data_id in labeled_dataset.ids_list:
            _, _, label = labeled_dataset._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        weights = self._cal_weights(num_each_class)
        self.weights = weights * self.num_cls
        return self.weights.data.cpu().numpy()

    def get_ema_weights(self, pseudo_label):
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=True).long()
        label_numpy = pseudo_label.data.cpu().numpy()
        num_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i].reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp

        cur_weights = self._cal_weights(num_each_class) * self.num_cls
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        return self.weights



class DiffDW:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(config.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        self.last_dice = cur_dice
        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
        cur_diff = torch.pow(cur_diff, 1/5)
        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        weights = weights / weights.max()
        return weights * self.num_cls





if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)



    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B  = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = kaiming_normal_init_weight(model_B)

    # Proxy modules: one ProjectionHead + one CompositionalSimilarityLoss per model
    # block_six feature: (B, 256, 8, 16, 16); variation_active defaults to True (R2)
    proj_head_A = ProjectionHead(in_channels=config.n_filters * 8, embedding_dim=args.embedding_dim, spatial_dims=3).cuda()
    cs_loss_A   = CompositionalSimilarityLoss(num_classes=config.num_cls, embedding_dim=args.embedding_dim,
                      use_variation=args.use_variation, num_variations=args.num_variations,
                      lambda_var=1.0, tau=args.tau_var, gamma=2.0, tau_r=0.8, lambda_r=1.0,
                      max_samples_per_class=args.max_samples_per_class).cuda()
    optimizer_proxy_A = optim.SGD(list(proj_head_A.parameters()) + list(cs_loss_A.parameters()),
                                  lr=args.base_lr, momentum=0.9, weight_decay=3e-5, nesterov=True)

    proj_head_B = ProjectionHead(in_channels=config.n_filters * 8, embedding_dim=args.embedding_dim, spatial_dims=3).cuda()
    cs_loss_B   = CompositionalSimilarityLoss(num_classes=config.num_cls, embedding_dim=args.embedding_dim,
                      use_variation=args.use_variation, num_variations=args.num_variations,
                      lambda_var=1.0, tau=args.tau_var, gamma=2.0, tau_r=0.8, lambda_r=1.0,
                      max_samples_per_class=args.max_samples_per_class).cuda()
    optimizer_proxy_B = optim.SGD(list(proj_head_B.parameters()) + list(cs_loss_B.parameters()),
                                  lr=args.base_lr, momentum=0.9, weight_decay=3e-5, nesterov=True)

    logging.info(f'Proxy: use_variation={args.use_variation}, lambda_cs={args.lambda_cs}, '
                 f'embedding_dim={args.embedding_dim}, num_variations={args.num_variations}, '
                 f'tau_var={args.tau_var}, max_samples_per_class={args.max_samples_per_class}, '
                 f'pseudo_proxy={args.pseudo_proxy}')

    # make loss function
    diffdw = DiffDW(config.num_cls, accumulate_iters=50)
    distdw = DistDW(config.num_cls, momentum=0.99)

    weight_A = diffdw.init_weights()
    weight_B = distdw.init_weights(labeled_loader.dataset)

    loss_func_A     = make_loss_function(args.sup_loss, weight_A)
    loss_func_B     = make_loss_function(args.sup_loss, weight_B)
    cps_loss_func_A = make_loss_function(args.cps_loss, weight_A)
    cps_loss_func_B = make_loss_function(args.cps_loss, weight_B)


    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model_A.train()
        model_B.train()
        proj_head_A.train()
        proj_head_B.train()

        # variation_warmup: keep variation_active=False until warmup period ends
        if args.use_variation and args.variation_warmup > 0:
            active = (epoch_num >= args.variation_warmup)
            cs_loss_A.variation_active = active
            cs_loss_B.variation_active = active

        # lambda_cs rampup: scale from 0 → target over rampup epochs
        if args.lambda_cs_rampup > 0:
            rampup_factor = min(1.0, epoch_num / args.lambda_cs_rampup)
        else:
            rampup_factor = 1.0
        effective_lambda_cs = args.lambda_cs * rampup_factor

        loss_cs_list = []
        for iteration_num, (batch_l, batch_u) in enumerate(tqdm(zip(labeled_loader, unlabeled_loader))):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()
            optimizer_proxy_A.zero_grad()
            optimizer_proxy_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            if args.mixed_precision:
                with autocast():
                    # return_features=True: feat is block_six (B,256,8,16,16), no detach (R6)
                    if args.lambda_cs > 0:
                        output_A, feat_A = model_A(image, return_features=True)
                        output_B, feat_B = model_B(image, return_features=True)
                    else:
                        output_A = model_A(image)
                        output_B = model_B(image)
                    del image

                    # sup (ce + dice)
                    output_A_l, output_A_u = output_A[:tmp_bs, ...], output_A[tmp_bs:, ...]
                    output_B_l, output_B_u = output_B[:tmp_bs, ...], output_B[tmp_bs:, ...]

                    # cps (ce only)
                    max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                    max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()

                    weight_A = diffdw.cal_weights(output_A_l.detach(), label_l.detach())
                    weight_B = distdw.get_ema_weights(output_B_u.detach())

                    loss_func_A.update_weight(weight_A)
                    loss_func_B.update_weight(weight_B)
                    cps_loss_func_A.update_weight(weight_A)
                    cps_loss_func_B.update_weight(weight_B)

                    loss_sup = loss_func_A(output_A_l, label_l) + loss_func_B(output_B_l, label_l)
                    loss_cps = cps_loss_func_A(output_A, max_B) + cps_loss_func_B(output_B, max_A)

                    if args.lambda_cs > 0:
                        # Proxy loss: labeled portion only; feat not detached so grad flows to backbone
                        emb_A = proj_head_A(feat_A[:tmp_bs])
                        emb_B = proj_head_B(feat_B[:tmp_bs])
                        loss_cs_A, stats_A = cs_loss_A(emb_A, label_l)
                        loss_cs_B, stats_B = cs_loss_B(emb_B, label_l)
                        loss_cs = loss_cs_A + loss_cs_B

                        # SAC: align proxy means with per-class embedding centroids (backbone detached via anchor)
                        if args.lambda_sac > 0:
                            loss_cs = loss_cs + args.lambda_sac * (
                                cs_loss_A.sac_loss(emb_A, label_l) +
                                cs_loss_B.sac_loss(emb_B, label_l)
                            )

                        # pseudo_proxy: unlabeled data with backbone DETACHED → only proj_head + proxy_dist update
                        if args.pseudo_proxy and epoch_num >= args.pseudo_proxy_warmup:
                            pp_elapsed = epoch_num - args.pseudo_proxy_warmup
                            pp_ramp = min(1.0, pp_elapsed / max(1, args.pseudo_proxy_w_rampup))
                            effective_pseudo_w = args.pseudo_proxy_w * pp_ramp
                            if effective_pseudo_w > 0:
                                emb_A_u = proj_head_A(feat_A[tmp_bs:].detach())
                                emb_B_u = proj_head_B(feat_B[tmp_bs:].detach())
                                loss_cs_A_u, _ = cs_loss_A(emb_A_u, max_B[tmp_bs:])
                                loss_cs_B_u, _ = cs_loss_B(emb_B_u, max_A[tmp_bs:])
                                loss_cs = loss_cs + effective_pseudo_w * (loss_cs_A_u + loss_cs_B_u)

                        loss = loss_sup + cps_w * loss_cps + effective_lambda_cs * loss_cs
                    else:
                        loss_cs = torch.tensor(0.0, device='cuda')
                        loss = loss_sup + cps_w * loss_cps

                # single backward; proxy optimizers only stepped when lambda_cs>0 (scaler
                # requires inf checks recorded, which only happen when params are used under autocast)
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                if args.lambda_cs > 0:
                    amp_grad_scaler.step(optimizer_proxy_A)
                    amp_grad_scaler.step(optimizer_proxy_B)
                amp_grad_scaler.update()

                # Gradient verification at step 5 (R6 / revision 1); encoder is a method not a module
                if args.lambda_cs > 0 and iteration_num == 5 and epoch_num == 0:
                    proxy_grad = cs_loss_A.proxy_dist.grad
                    backbone_grad = next(model_A.block_one.parameters()).grad
                    logging.info(f'[GradCheck] proxy_dist_A grad norm: '
                                 f'{proxy_grad.norm().item():.4e} (need >1e-3)')
                    logging.info(f'[GradCheck] backbone_A first param grad norm: '
                                 f'{backbone_grad.norm().item():.4e} (need >1e-3)')
                    if args.use_variation and cs_loss_A.variation_vectors is not None:
                        var_grad = cs_loss_A.variation_vectors.grad
                        var_norm = var_grad.norm().item() if var_grad is not None else float('nan')
                        logging.info(f'[GradCheck] variation_vectors_A grad norm: '
                                     f'{var_norm:.4e} (active={cs_loss_A.variation_active})')

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())
            loss_cs_list.append(loss_cs.item())

        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        writer.add_scalar('loss/cs', np.mean(loss_cs_list), epoch_num)
        writer.add_scalar('proxy/lambda_cs_eff', effective_lambda_cs, epoch_num)
        writer.add_scalar('proxy/lambda_sac', args.lambda_sac, epoch_num)
        if args.use_variation:
            writer.add_scalar('proxy/variation_active',
                              float(cs_loss_A.variation_active), epoch_num)
        # print(dict(zip([i for i in range(config.num_cls)] ,print_func(weight_A))))
        writer.add_scalars('class_weights/A', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_A))), epoch_num)
        writer.add_scalars('class_weights/B', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_B))), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | loss_cs : {np.mean(loss_cs_list):.4f}')
        # logging.info(f'     cps_w: {cps_w}')
        # if epoch_num>0:
        logging.info(f"     Class Weights A: {print_func(weight_A)}, lr: {get_lr(optimizer_A)}")
        logging.info(f"     Class Weights B: {print_func(weight_B)}")
        # logging.info(f"     Class Weights u: {print_func(weight_u)}")
        # lr_scheduler_A.step()
        # lr_scheduler_B.step()
        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_proxy_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_proxy_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        # print(optimizer_A.param_groups[0]['lr'])
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 10 == 0:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model_A.eval()
            model_B.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    output = (model_A(image) + model_B(image))/2.0
                    # output = model_B(image)
                    del image

                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)


                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            # '''
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'A': model_A.state_dict(),
                    'B': model_B.state_dict()
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    last_path = os.path.join(snapshot_path, 'ckpts/last_model.pth')
    torch.save({'A': model_A.state_dict(), 'B': model_B.state_dict()}, last_path)
    logging.info(f'saved last model to {last_path}')
    writer.close()
