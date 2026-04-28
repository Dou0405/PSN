import os
import os.path as osp
import json
import copy
import pickle
import logging
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import matplotlib as mpl # 需要导入 mpl 用于生成独立 colorbar

# 尝试导入 LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# 尝试导入 NPU 相关库
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

# 假设这些是你项目中的模块
from model import PSN
from API import *
from utils import * 
try:
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except ImportError:
    print("Warning: skimage not found, using simplified SSIM/PSNR")
    compare_ssim = None
    compare_psnr = None

# ==========================================
# 辅助模块：Loss 定义
# ==========================================

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.reshape(1, 1, 1, -1) * g.reshape(1, 1, -1, 1)

def ssim_loss(x, y, window_size=11, sigma=1.5):
    if x.dim() == 5:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        y = y.reshape(b * t, c, h, w)

    if x.shape[1] != y.shape[1]:
        if x.shape[1] > 1: x = x.mean(1, keepdim=True)
    
    channels = x.size(1)
    window = gaussian_window(window_size, sigma).to(x.device).type_as(x)
    window = window.repeat(channels, 1, 1, 1)
    
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=channels)
    
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x2 = F.conv2d(x**2, window, padding=window_size//2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y**2, window, padding=window_size//2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=channels) - mu_xy
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    
    return 1.0 - ssim_map.mean()

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        kernel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x, y):
        b, c, h, w = x.shape
        kx = self.kernel_x.expand(c, 1, 3, 3)
        ky = self.kernel_y.expand(c, 1, 3, 3)
        pred_grad_x = F.conv2d(x, kx, padding=1, groups=c)
        pred_grad_y = F.conv2d(x, ky, padding=1, groups=c)
        target_grad_x = F.conv2d(y, kx, padding=1, groups=c)
        target_grad_y = F.conv2d(y, ky, padding=1, groups=c)
        loss = F.l1_loss(torch.abs(pred_grad_x) + torch.abs(pred_grad_y),
                         torch.abs(target_grad_x) + torch.abs(target_grad_y))
        return loss

class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        loss = self.criterion(pred_fft.real, target_fft.real) + \
               self.criterion(pred_fft.imag, target_fft.imag)
        return loss

class VGGLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 17, 26]):
        super(VGGLoss, self).__init__()
        try:
            vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        except:
            vgg = torchvision.models.vgg19(pretrained=True)
        self.vgg_layers = vgg.features
        self.layer_ids = layer_ids
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            if next(self.vgg_layers.parameters()).device != x.device:
                self.vgg_layers = self.vgg_layers.to(x.device)
        
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += F.l1_loss(x, y)
            if i >= max(self.layer_ids):
                break
        return loss

class HybridLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        
        if self.weights.get('fft', 0) > 0:
            self.fft_loss_fn = FFTLoss()
        if self.weights.get('grad', 0) > 0:
            self.grad_loss_fn = GradientLoss()
        if self.weights.get('vgg', 0) > 0:
            self.vgg_loss_fn = VGGLoss()
        if self.weights.get('lpips', 0) > 0 and LPIPS_AVAILABLE:
            self.lpips_loss_fn = lpips.LPIPS(net='alex')
            for param in self.lpips_loss_fn.parameters():
                param.requires_grad = False

    def forward(self, pred, target, mask=None):
        loss_dict = {}
        if pred.dim() == 5:
            b, t, c, h, w = pred.shape
            pred_2d = pred.reshape(b * t, c, h, w)
            target_2d = target.reshape(b * t, c, h, w)
            mask_2d = mask.reshape(b * t, c, h, w) if (mask is not None and mask.dim() == 5) else mask
        else:
            pred_2d, target_2d, mask_2d = pred, target, mask

        w_l1 = self.weights.get('l1', 1.0)
        if w_l1 > 0:
            if mask_2d is not None:
                diff = torch.abs(pred_2d - target_2d)
                pixel_loss = (diff * mask_2d).sum() / (mask_2d.sum() + 1e-8)
            else:
                pixel_loss = self.l1(pred_2d, target_2d)
            loss_dict['l1'] = pixel_loss

        w_ssim = self.weights.get('ssim', 0.0)
        if w_ssim > 0:
            loss_dict['ssim'] = ssim_loss(pred_2d, target_2d)

        w_fft = self.weights.get('fft', 0.0)
        if w_fft > 0:
            loss_dict['fft'] = self.fft_loss_fn(pred_2d, target_2d)

        w_grad = self.weights.get('grad', 0.0)
        if w_grad > 0:
            loss_dict['grad'] = self.grad_loss_fn(pred_2d, target_2d)

        w_vgg = self.weights.get('vgg', 0.0)
        if w_vgg > 0:
            loss_dict['vgg'] = self.vgg_loss_fn(pred_2d, target_2d)

        w_lpips = self.weights.get('lpips', 0.0)
        if w_lpips > 0 and LPIPS_AVAILABLE:
            p_norm = pred_2d * 2 - 1 
            t_norm = target_2d * 2 - 1
            if p_norm.shape[1] == 1: 
                p_norm = p_norm.repeat(1, 3, 1, 1)
                t_norm = t_norm.repeat(1, 3, 1, 1)
            if next(self.lpips_loss_fn.parameters()).device != pred.device:
                self.lpips_loss_fn.to(pred.device)
            loss_dict['lpips'] = self.lpips_loss_fn(p_norm, t_norm).mean()

        total_loss = 0.0
        for k, v in loss_dict.items():
            total_loss += v * self.weights.get(k, 0.0)
        return total_loss, loss_dict

# ==========================================
# 工具函数：指标计算
# ==========================================

def calculate_numpy_metrics(pred_np, true_np, data_mean=None, data_std=None, return_individual=False):
    if data_mean is not None and data_std is not None:
        mean = np.array(data_mean).reshape(1, 1, -1, 1, 1)
        std = np.array(data_std).reshape(1, 1, -1, 1, 1)
        pred_real = pred_np * std + mean
        true_real = true_np * std + mean
    else:
        pred_real = pred_np
        true_real = true_np

    B, T, C, H, W = pred_real.shape
    
    # 计算误差
    err = pred_real - true_real
    abs_err = np.abs(err)
    sq_err = err ** 2

    # 计算相对误差 (增加 epsilon 防止除零)
    denom = np.abs(true_real) + 1e-5
    rel_abs_err = abs_err / denom
    rel_sq_err = sq_err / (denom ** 2)
    
    # ------------------ 总和计算 (按通道统计) ------------------
    # 绝对指标
    sum_se = np.sum(sq_err, axis=(0, 1, 3, 4))   # Sum Squared Error [C]
    sum_ae = np.sum(abs_err, axis=(0, 1, 3, 4))  # Sum Absolute Error [C]
    
    # 相对指标
    sum_rse = np.sum(rel_sq_err, axis=(0, 1, 3, 4)) # Sum Relative Squared Error [C]
    sum_rae = np.sum(rel_abs_err, axis=(0, 1, 3, 4)) # Sum Relative Absolute Error [C]
    
    channel_pixels = B * T * H * W
    frames_count = B * T

    ssim_accum = np.zeros(C)
    psnr_accum = np.zeros(C)

    # ------------------ 逐样本列表 (用于单独打印) ------------------
    individual_metrics = [] 

    for b in range(B):
        # 记录该样本的相对MAE总和 (用于计算样本级平均)
        sample_rae_sum = np.sum(rel_abs_err[b], axis=(0, 2, 3)) # [C]
        
        sample_ssim = np.zeros(C)
        sample_psnr = np.zeros(C)
        
        for t in range(T):
            for c in range(C):
                p_img = pred_real[b, t, c]
                t_img = true_real[b, t, c]
                
                data_range = t_img.max() - t_img.min()
                if data_range == 0: data_range = 1.0
                
                # SSIM
                val_ssim = 0
                if compare_ssim:
                    try:
                        val_ssim = compare_ssim(t_img, p_img, data_range=data_range)
                    except ValueError:
                        val_ssim = compare_ssim(t_img, p_img, data_range=data_range, win_size=3)
                    ssim_accum[c] += val_ssim
                    sample_ssim[c] += val_ssim
                
                # PSNR
                val_psnr = 0
                if compare_psnr:
                    val_psnr = compare_psnr(t_img, p_img, data_range=data_range)
                    psnr_accum[c] += val_psnr
                    sample_psnr[c] += val_psnr
        
        sample_pixels = T * H * W 
        sample_frames = T
        
        m_dict = {}
        for c in range(C):
            # 存储单样本的 Relative MAE, SSIM, PSNR
            m_dict[f'rmae_c{c}'] = sample_rae_sum[c] / (sample_pixels + 1e-9)
            m_dict[f'ssim_c{c}'] = sample_ssim[c] / (sample_frames + 1e-9)
            m_dict[f'psnr_c{c}'] = sample_psnr[c] / (sample_frames + 1e-9)

        individual_metrics.append(m_dict)

    # 返回所有累加器
    total_metrics = (sum_se, sum_ae, sum_rse, sum_rae, ssim_accum, psnr_accum, channel_pixels, frames_count)

    if return_individual:
        return total_metrics, individual_metrics
    else:
        return total_metrics

# ==========================================
# Exp 类
# ==========================================

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))
        
        self._get_data()
        self._build_model()
        self._select_optimizer()
        self._select_criterion()
        
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model = self.ema_model.to(self.device)
        print_log(f"EMA模型已迁移到{self.device}")
        
        self.update_ema_every = 10
        self.step_start_ema = 2000
        self.step = 0

        self._debug_logged = False
        self.t_sample = [0.1] * 10
        
        self.best_vali_ssim = 0.0

        self._load()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def _acquire_device(self):
        if self.args.use_npu:
            os.environ["ASCEND_DEVICE_ID"] = str(self.args.npu)
            try:
                if torch.npu.is_available():
                    device = torch.device(f'npu:{self.args.npu}') 
                    print_log(f'Use NPU: {self.args.npu}')
                else:
                    raise RuntimeError("NPU unavailable")
            except:
                print_log(f'NPU {self.args.npu}不可用，自动降级为CPU')
                device = torch.device('cpu')
        elif self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{0}')
                print_log(f'Use GPU: {self.args.gpu}')
            else:
                print_log(f'GPU {self.args.gpu}不可用，自动降级为CPU')
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        set_seed(self.args.seed)
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        log_file_path = osp.join(self.path, 'log.txt')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        if self.args.test is not None:
            logging.info(f"日志同步输出到 {log_file_path} 和控制台")

    def _build_model(self):
        args = self.args
        self.model = PSN(
            tuple(args.in_shape),
            tuple(args.out_shape),
            args.hid_S,
            args.hid_T,
            args.N_S,
            args.N_T,
            use_physical_prior=getattr(args, 'use_physical_prior', False)
        )
        self.model = self.model.to(self.device)
        print_log(f"模型已迁移到{self.device}")

    def _get_data(self):
        config = dict(self.args.__dict__)
        dataname = config.pop('dataname', 'aifund')
        batch_size = config.pop('batch_size', 1)
        val_batch_size = config.pop('val_batch_size', 1)
        data_root = config.pop('data_root', './data/')
        num_workers = config.pop('num_workers', 0)

        kwargs = config
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = \
            load_data(dataname, batch_size, val_batch_size, data_root, num_workers, **kwargs)

        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader
        if self.train_loader is not None:
            print_log(f"Data Loaded. Train batches: {len(self.train_loader)}")

    def _select_optimizer(self):
        if self.args.test:
            return None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if not hasattr(self, 'train_loader') or self.train_loader is None:
             raise AttributeError("train_loader not found or empty.")
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), pct_start=0.0, epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        weights = {
            'l1':    getattr(self.args, 'l1_weight', 1.0),
            'ssim':  getattr(self.args, 'ssim_weight', 0.0),
            'fft':   getattr(self.args, 'fft_weight', 0.0), 
            'grad':  getattr(self.args, 'grad_weight', 0.0),
            'vgg':   getattr(self.args, 'vgg_weight', 0.0),
            'lpips': getattr(self.args, 'lpips_weight', 0.0)
        }
        self.criterion = HybridLoss(weights)
        self.criterion = self.criterion.to(self.device)
        
        log_str = "Using HybridLoss with weights: " + ", ".join([f"{k}:{v}" for k, v in weights.items() if v > 0])
        print_log(log_str)

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        torch.save(self.ema_model.state_dict(), os.path.join(
            self.checkpoints_path, name + '_ema.pth'))
        state = self.scheduler.state_dict()
        with open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb') as fw:
            pickle.dump(state, fw)

    def _load(self):
        model_path = os.path.join(self.path, 'checkpoint.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.ema_model.load_state_dict(state_dict)
            self.ema_model.eval()
            print_log(f'Load model from {model_path} (设备：{self.device})')
        else:
            print_log(f'No checkpoint found at {model_path}')

    def train(self, args):
            config = args.__dict__
            recorder = Recorder(verbose=True)

            for epoch in range(config['epochs']):
                train_loss = []
                self.model.train()
                train_pbar = tqdm(self.train_loader, ncols=140) 
                
                for batch in train_pbar:
                    if getattr(self.args, 'use_mask', False):
                        try:
                            batch_x, batch_y, mask_batch = batch
                        except Exception:
                            batch_x, batch_y = batch
                            mask_batch = None
                    else:
                        batch_x, batch_y = batch
                        mask_batch = None

                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    if mask_batch is not None:
                        mask_batch = mask_batch.to(self.device)
                    
                    pred_list = []
                    for times in range(batch_y.shape[1]):
                        self.optimizer.zero_grad()
                        t = torch.tensor(times*100, dtype=torch.float32, device=self.device).repeat(batch_x.shape[0])
                        
                        pred_y = self.model(batch_x, pred_list, t)
                        
                        if pred_y.dim() == 4:
                            pred_y = pred_y.unsqueeze(1)
                        
                        target = batch_y[:, times:times+1, :, :, :] 
                        
                        mask_weight = None
                        if mask_batch is not None:
                            weights = mask_to_weight(mask_batch, alpha=getattr(self.args, 'alpha', 0.1))
                            if weights.dim() == 4:
                                weights = weights.unsqueeze(1)
                            mask_weight = weights.expand_as(pred_y)
                        
                        loss, loss_dict = self.criterion(pred_y, target, mask=mask_weight)
                        
                        loss_cpu = loss.item() if not torch.isnan(loss) else 0.0
                        train_loss.append(loss_cpu)
                        
                        desc_str = f'Loss:{loss_cpu:.4f}'
                        for k, v in loss_dict.items():
                            if self.criterion.weights.get(k, 0) > 0:
                                val = v.item() if hasattr(v, 'item') else v
                                desc_str += f' {k}:{val:.4f}'
                        train_pbar.set_description(desc_str)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        pred_list.append(pred_y.squeeze(1).detach()) 
                        self.optimizer.step()
                    
                    self.scheduler.step()

                    if self.step % self.update_ema_every == 0:
                        self.step_ema()
                    self.step += 1

                train_loss = np.average(train_loss)

                if epoch % args.log_step == 0:
                    with torch.no_grad():
                        vali_loss, mse_avg, ssim_avg, psnr_avg = self.vali(self.vali_loader)

                        if ssim_avg > self.best_vali_ssim:
                            self.best_vali_ssim = ssim_avg
                            logging.info(f"Epoch {epoch+1}: New best SSIM model found (SSIM: {ssim_avg:.4f})!")
                            self._save(name='best_model')
                            self.save_inference_results(self.vali_loader, folder_name='results_best')
                        
                        if epoch % (args.log_step * 10) == 0:
                            self._save(name=str(epoch))
                    
                    logging.info(
                        "Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f} | "
                        "Vali Avg MSE: {3:.5f} SSIM: {4:.4f} PSNR: {5:.4f}".format(
                            epoch + 1, train_loss, vali_loss, mse_avg, ssim_avg, psnr_avg
                        )
                    )
                    
                    recorder(vali_loss, self.model, self.path)

            best_model_path = self.path + '/' + 'checkpoint.pth'
            if os.path.exists(best_model_path):
                state_dict = torch.load(best_model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            return self.model

    def vali(self, vali_loader):
        self.model.eval()
        self.ema_model.eval()
        
        total_loss_list = []
        
        acc_se = None 
        acc_ae = None
        acc_rse = None
        acc_rae = None
        acc_ssim = None
        acc_psnr = None
        acc_frames = 0
        channel_pixels = 0
        
        vali_pbar = tqdm(vali_loader, ncols=100)
        
        logging.info("\n" + "="*20 + " Validation Individual Results " + "="*20)

        with torch.no_grad():
            for i, batch in enumerate(vali_pbar):
                output_frame_names = None
                if len(batch) >= 3:
                    batch_x = batch[0]
                    batch_y = batch[1]
                    output_frame_names = batch[2]
                else:
                    batch_x, batch_y = batch
                
                if i * batch_x.shape[0] > 1000: break 

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_list = []
                B, T, C, H, W = batch_y.shape
                
                if acc_se is None:
                    acc_se = np.zeros(C)
                    acc_ae = np.zeros(C)
                    acc_rse = np.zeros(C)
                    acc_rae = np.zeros(C)
                    acc_ssim = np.zeros(C)
                    acc_psnr = np.zeros(C)
                
                for timestep in range(T):
                    t = torch.tensor(timestep*100, dtype=torch.float32, device=self.device).repeat(B)
                    pred_y = self.ema_model(batch_x, pred_list, t)
                    pred_list.append(pred_y)

                pred_y = torch.stack(pred_list, dim=1)
                
                loss, _ = self.criterion(pred_y, batch_y)
                total_loss_list.append(loss.item() if not torch.isnan(loss) else 0.0)
                
                pred_np = pred_y.detach().cpu().numpy()
                true_np = batch_y.detach().cpu().numpy()
                
                metrics_res, ind_metrics_list = calculate_numpy_metrics(
                    pred_np, true_np, self.data_mean, self.data_std, return_individual=True
                )
                
                sum_se, sum_ae, sum_rse, sum_rae, ssim_val, psnr_val, pixels, frames = metrics_res
                
                acc_se += sum_se
                acc_ae += sum_ae
                acc_rse += sum_rse
                acc_rae += sum_rae
                acc_ssim += ssim_val
                acc_psnr += psnr_val
                acc_frames += frames
                channel_pixels += pixels 
                
                for b in range(B):
                    sample_name = f"Batch{i}_Sample{b}"
                    if output_frame_names is not None:
                        try:
                            first_frame_path = Path(output_frame_names[b][0])
                            sample_name = first_frame_path.parent.parent.name
                        except: pass
                    
                    m = ind_metrics_list[b]
                    log_str = f"[Vali] {sample_name}: "
                    for c in range(C):
                        if C > 1:
                            c_name = "PEEQ" if c==0 else ("MISES" if c==1 else f"Ch{c}")
                        else:
                            c_name = self.config.get('data_type', 'PEEQ').upper()
                            
                        log_str += f"| {c_name} [RMAE:{m[f'rmae_c{c}']:.5f} SSIM:{m[f'ssim_c{c}']:.4f} PSNR:{m[f'psnr_c{c}']:.4f}] "
                    logging.info(log_str)

                vali_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))

        avg_loss = np.average(total_loss_list)
        
        avg_mse_all = np.sum(acc_se) / (channel_pixels * C + 1e-9)
        avg_ssim_all = np.sum(acc_ssim) / (acc_frames * C + 1e-9)
        avg_psnr_all = np.sum(acc_psnr) / (acc_frames * C + 1e-9)

        logging.info("="*60)
        logging.info(f"Validation Summary:")
        if acc_se is not None:
            num_channels = len(acc_se)
            for c in range(num_channels):
                c_abs_mse = acc_se[c] / (channel_pixels + 1e-9)
                c_abs_mae = acc_ae[c] / (channel_pixels + 1e-9)
                c_rel_mse = acc_rse[c] / (channel_pixels + 1e-9)
                c_rel_mae = acc_rae[c] / (channel_pixels + 1e-9)
                
                c_ssim = acc_ssim[c] / (acc_frames + 1e-9)
                c_psnr = acc_psnr[c] / (acc_frames + 1e-9)
                
                if num_channels > 1:
                    c_name = "PEEQ" if c==0 else ("MISES" if c==1 else f"Channel {c}")
                else:
                    c_name = self.config.get('data_type', 'PEEQ').upper()
                
                logging.info(f"  > {c_name:<6} : "
                             f"AbsMSE={c_abs_mse:.5f}, AbsMAE={c_abs_mae:.5f}, "
                             f"RelMSE={c_rel_mse:.5f}, RelMAE={c_rel_mae:.5f}, "
                             f"SSIM={c_ssim:.4f}, PSNR={c_psnr:.4f}")

        logging.info("="*60 + "\n")

        self.model.train()
        return avg_loss, avg_mse_all, avg_ssim_all, avg_psnr_all

    def test(self, args):
            self.model.eval()
            self.ema_model.eval()
            
            if hasattr(self, 'train_loader'): del self.train_loader
            if hasattr(self, 'vali_loader'): del self.vali_loader
            
            self.save_inference_results(self.test_loader, folder_name='results')

            test_bar = tqdm(self.test_loader, ncols=120, desc='Calculating Test Metrics')
            
            acc_se = None
            acc_ae = None
            acc_rse = None
            acc_rae = None
            acc_ssim = None
            acc_psnr = None
            acc_frames = 0
            channel_pixels = 0

            logging.info("\n" + "="*20 + " Test Individual Results " + "="*20)

            with torch.no_grad():
                for i, batch in enumerate(test_bar):
                    output_frame_names = None
                    if len(batch) >= 3:
                        batch_x = batch[0]
                        batch_y = batch[1]
                        output_frame_names = batch[2]
                    else:
                        batch_x, batch_y = batch
                        
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_list = []
                    B, T, C, H, W = batch_y.shape
                    
                    if acc_se is None:
                        acc_se = np.zeros(C)
                        acc_ae = np.zeros(C)
                        acc_rse = np.zeros(C)
                        acc_rae = np.zeros(C)
                        acc_ssim = np.zeros(C)
                        acc_psnr = np.zeros(C)
                    
                    for timestep in range(T):
                        t = torch.tensor(timestep*100, dtype=torch.float32, device=self.device).repeat(B)
                        pred_y = self.ema_model(batch_x, pred_list, t)
                        pred_list.append(pred_y)

                    pred_y = torch.stack(pred_list, dim=1)
                    pred_np = pred_y.detach().cpu().numpy()
                    true_np = batch_y.detach().cpu().numpy()
                    
                    metrics_res, ind_metrics_list = calculate_numpy_metrics(
                        pred_np, true_np, self.test_loader.dataset.mean, self.test_loader.dataset.std, return_individual=True
                    )
                    sum_se, sum_ae, sum_rse, sum_rae, ssim_val, psnr_val, pixels, frames = metrics_res
                    
                    acc_se += sum_se
                    acc_ae += sum_ae
                    acc_rse += sum_rse
                    acc_rae += sum_rae
                    acc_ssim += ssim_val
                    acc_psnr += psnr_val
                    acc_frames += frames
                    channel_pixels += pixels

                    for b in range(B):
                        sample_name = f"Batch{i}_Sample{b}"
                        if output_frame_names is not None:
                            try:
                                first_frame_path = Path(output_frame_names[b][0])
                                sample_name = first_frame_path.parent.parent.name
                            except: pass
                        
                        m = ind_metrics_list[b]
                        log_str = f"[Test] {sample_name}: "
                        for c in range(C):
                            if C > 1:
                                c_name = "PEEQ" if c==0 else ("MISES" if c==1 else f"Ch{c}")
                            else:
                                c_name = self.config.get('data_type', 'PEEQ').upper()
                            log_str += f"| {c_name} [RMAE:{m[f'rmae_c{c}']:.5f} SSIM:{m[f'ssim_c{c}']:.4f} PSNR:{m[f'psnr_c{c}']:.4f}] "
                        logging.info(log_str)

            final_mse_avg = np.sum(acc_se) / (channel_pixels * C + 1e-9)

            logging.info("="*60 + "\n")
            logging.info("-" * 30 + " TEST REPORT (BREAKDOWN) " + "-" * 30)
            
            if acc_se is not None:
                for c in range(len(acc_se)):
                    c_abs_mse = acc_se[c] / (channel_pixels + 1e-9)
                    c_abs_mae = acc_ae[c] / (channel_pixels + 1e-9)
                    c_rel_mse = acc_rse[c] / (channel_pixels + 1e-9)
                    c_rel_mae = acc_rae[c] / (channel_pixels + 1e-9)
                    c_ssim = acc_ssim[c] / (acc_frames + 1e-9)
                    c_psnr = acc_psnr[c] / (acc_frames + 1e-9)
                    
                    if len(acc_se) > 1:
                        c_name = "PEEQ" if c==0 else ("MISES" if c==1 else f"Channel {c}")
                    else:
                        c_name = self.config.get('data_type', 'PEEQ').upper()
                    
                    logging.info(f"{c_name:<8} | AbsMSE: {c_abs_mse:.6f} | AbsMAE: {c_abs_mae:.6f} | "
                                 f"RelMSE: {c_rel_mse:.6f} | RelMAE: {c_rel_mae:.6f} | "
                                 f"SSIM: {c_ssim:.4f} | PSNR: {c_psnr:.4f}")
            
            logging.info("-" * 73)
            return final_mse_avg

    def save_inference_results(self, loader, folder_name='results_best'):
            self.ema_model.eval()
            save_folder = os.path.join(self.path, folder_name)
            os.makedirs(save_folder, exist_ok=True)
            
            saliency_folder = os.path.join(save_folder, 'saliency_maps')
            os.makedirs(saliency_folder, exist_ok=True)

            print_log(f'Saving visual results to {save_folder} ...')
            vis_bar = tqdm(loader, ncols=120)
            
            if self.data_mean is not None:
                d_mean = np.array(self.data_mean).reshape(1, 1, -1, 1, 1)
                d_std = np.array(self.data_std).reshape(1, 1, -1, 1, 1)
            else:
                d_mean, d_std = 0, 1

            with torch.no_grad():
                for i, batch in enumerate(vis_bar):
                    output_frame_names = None
                    masks = None
                    if len(batch) == 3:
                        if isinstance(batch[2], list):
                            batch_x, batch_y, output_frame_names = batch
                        else:
                            batch_x, batch_y, masks = batch
                    elif len(batch) == 4:
                        batch_x, batch_y, output_frame_names, masks = batch
                    else:
                        batch_x, batch_y = batch

                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_list = []
                    B, T, C, H, W = batch_y.shape
                    
                    for timestep in range(T):
                        t = torch.tensor(timestep*100, dtype=torch.float32, device=self.device).repeat(B)
                        pred_y = self.ema_model(batch_x, pred_list, t)
                        pred_list.append(pred_y)

                    pred_y = torch.stack(pred_list, dim=1)
                    
                    pred_np = pred_y.detach().cpu().numpy()
                    gt_np = batch_y.detach().cpu().numpy()
                    pred_real = pred_np * d_std + d_mean
                    gt_real = gt_np * d_std + d_mean
                    
                    mask_np = None
                    if masks is not None:
                        mask_np = masks.detach().cpu().numpy()

                    for b in range(B):
                        seq_name = f"batch_{i}_sample_{b}"
                        if output_frame_names is not None:
                            try:
                                first_frame_path = Path(output_frame_names[b][0])
                                seq_name = first_frame_path.parent.parent.name
                            except: pass
                        
                        current_sample_root = os.path.join(save_folder, seq_name)
                        
                        # ---------------------------------------------------------
                        # [Add] Saliency Map 保存逻辑 (颗粒置黑)
                        # ---------------------------------------------------------
                        if hasattr(self.ema_model, 'visual_map') and self.ema_model.visual_map is not None:
                            try:
                                visual_map = self.ema_model.visual_map.detach().cpu().numpy()
                                # [B, T_in, 1, H, W] -> 取 Sample b, 第0帧, 第0通道 -> [H, W]
                                saliency_img = visual_map[b, 0, 0] 
                                
                                # 如果有 Mask，将颗粒部分 (数值为0的部分) 置为黑色 (0.0)
                                if mask_np is not None:
                                    m_raw = mask_np[b, 0] # [H, W]
                                    is_particle = (m_raw < 0.5)
                                    saliency_img[is_particle] = 0.0
                                
                                saliency_img_uint8 = (saliency_img * 255).astype(np.uint8)
                                Image.fromarray(saliency_img_uint8).save(os.path.join(saliency_folder, f"{seq_name}_saliency.png"))
                            except Exception as e:
                                pass

                        # [Step 1] 预计算全局量程
                        channel_limits = {} 
                        
                        particle_mask_dilated = None
                        if mask_np is not None:
                            m_raw = mask_np[b, 0] 
                            is_matrix = (m_raw > 0.5) 
                            
                            particle_mask = (m_raw < 0.5)
                            particle_mask_dilated = binary_dilation(particle_mask, iterations=1)
                            # particle_mask_dilated = particle_mask
                        else:
                            is_matrix = np.ones((H, W), dtype=bool)

                        for ch in range(C):
                            # 判断是否为 Mises 通道
                            is_mises_channel = False
                            if C > 1:
                                if ch == 1: is_mises_channel = True
                            else:
                                data_type = self.config.get('data_type', 'peeq').lower()
                                if 'mises' in data_type: is_mises_channel = True
                            
                            seq_gt = gt_real[b, :, ch]
                            seq_pred = pred_real[b, :, ch]
                            seq_err = np.abs(seq_gt - seq_pred)
                            
                            # --- Value Range Setting ---
                            matrix_vals = seq_gt[:, is_matrix]
                            
                            if matrix_vals.size > 0:
                                if is_mises_channel:
                                    g_min = np.percentile(matrix_vals, 0)
                                    g_max = np.percentile(matrix_vals, self.args.mises_percentile)
                                else:
                                    g_min = np.percentile(matrix_vals, 0)
                                    g_max = np.percentile(matrix_vals, self.args.peeq_percentile)
                            else:
                                g_min, g_max = seq_gt.min(), seq_gt.max()
                            
                            if g_max <= g_min: g_max = g_min + 1e-5
                            
                            # --- Error Range Setting ---
                            # [Error Map] Always use Absolute Max (No truncation)
                            matrix_errs = seq_err[:, is_matrix]
                            if matrix_errs.size > 0:
                                e_max = matrix_errs.max() 
                            else:
                                e_max = seq_err.max()
                            
                            if e_max < 1e-6: e_max = 1e-5
                            
                            channel_limits[ch] = {'val': (g_min, g_max), 'err': (0, e_max)}

                        # [Step 2] 绘图与保存
                        for t in range(T):
                            for ch in range(C):
                                if C > 1:
                                    ch_name = "peeq" if ch == 0 else ("mises" if ch == 1 else f"ch{ch}")
                                else:
                                    ch_name = self.config.get('data_type', 'peeq').lower()
                                
                                combined_save_path = os.path.join(current_sample_root, ch_name)
                                os.makedirs(combined_save_path, exist_ok=True)
                                
                                gt_save_path = os.path.join(current_sample_root, f"{ch_name}_gt")
                                pred_save_path = os.path.join(current_sample_root, f"{ch_name}_pred")
                                os.makedirs(gt_save_path, exist_ok=True)
                                os.makedirs(pred_save_path, exist_ok=True)
                                
                                img_gt = gt_real[b, t, ch]
                                img_pred = pred_real[b, t, ch]
                                img_err = np.abs(img_gt - img_pred)
                                
                                limits = channel_limits[ch]
                                vmin, vmax = limits['val']
                                emin, emax = limits['err']
                                
                                file_name_stem = f"t_{t}"
                                if output_frame_names is not None:
                                    try:
                                        frame_path = Path(output_frame_names[b][t])
                                        base = frame_path.stem 
                                        if base.lower().startswith('peeq'): base = base[4:]
                                        elif base.lower().startswith('mises'): base = base[5:]
                                        file_name_stem = base
                                    except: pass
                                
                                # A. 绘制组合图 (GT, Pred, Error)
                                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                                
                                if particle_mask_dilated is not None:
                                    # GT/Pred: 白色遮挡
                                    img_gt_vis = np.ma.masked_where(particle_mask_dilated, img_gt)
                                    img_pred_vis = np.ma.masked_where(particle_mask_dilated, img_pred)
                                    # Error: 强制置 0 (显示为蓝色/底色)
                                    img_err_vis = img_err.copy()
                                    img_err_vis[particle_mask_dilated] = 0.0
                                else:
                                    img_gt_vis, img_pred_vis, img_err_vis = img_gt, img_pred, img_err

                                cmap = copy.copy(plt.get_cmap("jet"))
                                cmap.set_bad(color='white')
                                
                                # GT
                                im1 = axes[0].imshow(img_gt_vis, cmap=cmap, vmin=vmin, vmax=vmax)
                                axes[0].set_title(f"GT ({ch_name})")
                                axes[0].axis('off')
                                plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                                # Pred
                                im2 = axes[1].imshow(img_pred_vis, cmap=cmap, vmin=vmin, vmax=vmax)
                                axes[1].set_title(f"Pred ({ch_name})")
                                axes[1].axis('off')
                                plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                                
                                # Error
                                im3 = axes[2].imshow(img_err_vis, cmap=cmap, vmin=emin, vmax=emax)
                                axes[2].set_title(f"Error Map")
                                axes[2].axis('off')
                                plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
                                
                                plt.tight_layout()
                                plt.savefig(os.path.join(combined_save_path, f"{file_name_stem}.png"), dpi=100)
                                plt.close(fig)
                                
                                # B. 保存独立纯净图像
                                def save_clean_image(img_data, save_path, v_min, v_max, mask):
                                    norm_data = (img_data - v_min) / (v_max - v_min + 1e-9)
                                    norm_data = np.clip(norm_data, 0, 1)
                                    colormap = plt.get_cmap("jet")
                                    colored = colormap(norm_data) # RGBA
                                    if mask is not None:
                                        colored[mask] = [1.0, 1.0, 1.0, 1.0]
                                    img_uint8 = (colored * 255).astype(np.uint8)
                                    Image.fromarray(img_uint8).save(save_path)

                                save_clean_image(img_gt, 
                                                os.path.join(gt_save_path, f"{file_name_stem}.png"), 
                                                vmin, vmax, particle_mask_dilated)
                                
                                save_clean_image(img_pred, 
                                                os.path.join(pred_save_path, f"{file_name_stem}.png"), 
                                                vmin, vmax, particle_mask_dilated)

            self.model.train()


    def save_inference_results(self, loader, folder_name='results_best'):
            self.ema_model.eval()
            save_folder = os.path.join(self.path, folder_name)
            os.makedirs(save_folder, exist_ok=True)
            
            saliency_folder = os.path.join(save_folder, 'saliency_maps')
            os.makedirs(saliency_folder, exist_ok=True)

            print_log(f'Saving visual results to {save_folder} ...')
            vis_bar = tqdm(loader, ncols=120)
            
            if self.data_mean is not None:
                d_mean = np.array(self.data_mean).reshape(1, 1, -1, 1, 1)
                d_std = np.array(self.data_std).reshape(1, 1, -1, 1, 1)
            else:
                d_mean, d_std = 0, 1

            with torch.no_grad():
                for i, batch in enumerate(vis_bar):
                    # --- 数据解包 ---
                    output_frame_names = None
                    masks = None
                    if len(batch) == 3:
                        if isinstance(batch[2], list):
                            batch_x, batch_y, output_frame_names = batch
                        else:
                            batch_x, batch_y, masks = batch
                    elif len(batch) == 4:
                        batch_x, batch_y, output_frame_names, masks = batch
                    else:
                        batch_x, batch_y = batch

                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_list = []
                    B, T, C, H, W = batch_y.shape
                    
                    # --- 模型推理 ---
                    for timestep in range(T):
                        t_tensor = torch.tensor(timestep*100, dtype=torch.float32, device=self.device).repeat(B)
                        pred_y = self.ema_model(batch_x, pred_list, t_tensor)
                        pred_list.append(pred_y)

                    pred_y = torch.stack(pred_list, dim=1)
                    
                    # 反归一化
                    pred_np = pred_y.detach().cpu().numpy()
                    gt_np = batch_y.detach().cpu().numpy()
                    pred_real = pred_np * d_std + d_mean
                    gt_real = gt_np * d_std + d_mean
                    
                    mask_np = None
                    if masks is not None:
                        mask_np = masks.detach().cpu().numpy()

                    for b in range(B):
                        # 获取样本名
                        seq_name = f"batch_{i}_sample_{b}"
                        if output_frame_names is not None:
                            try:
                                first_frame_path = Path(output_frame_names[b][0])
                                seq_name = first_frame_path.parent.parent.name
                            except: pass
                        
                        current_sample_root = os.path.join(save_folder, seq_name)

                        # --- 0. Saliency Map (保持原样) ---
                        if hasattr(self.ema_model, 'visual_map') and self.ema_model.visual_map is not None:
                            try:
                                visual_map = self.ema_model.visual_map.detach().cpu().numpy()
                                saliency_img = visual_map[b, 0, 0] 
                                if mask_np is not None:
                                    m_raw = mask_np[b, 0]
                                    is_particle = (m_raw < 0.5)
                                    saliency_img[is_particle] = 0.0
                                saliency_img_uint8 = (saliency_img * 255).astype(np.uint8)
                                Image.fromarray(saliency_img_uint8).save(os.path.join(saliency_folder, f"{seq_name}_saliency.png"))
                            except Exception: pass
                        
                        # --- 1. 计算全局量程 (基于 GT) ---
                        channel_limits = {} 
                        particle_mask_dilated = None
                        
                        if mask_np is not None:
                            m_raw = mask_np[b, 0] 
                            is_matrix = (m_raw > 0.5) 
                            particle_mask = (m_raw < 0.5)
                            particle_mask_dilated = binary_dilation(particle_mask, iterations=1)
                        else:
                            is_matrix = np.ones((H, W), dtype=bool)

                        for ch in range(C):
                            is_mises_channel = False
                            if C > 1:
                                if ch == 1: is_mises_channel = True
                            else:
                                data_type = self.config.get('data_type', 'peeq').lower()
                                if 'mises' in data_type: is_mises_channel = True
                            
                            seq_gt = gt_real[b, :, ch]
                            matrix_vals = seq_gt[:, is_matrix]
                            
                            if matrix_vals.size > 0:
                                if is_mises_channel:
                                    g_min = np.percentile(matrix_vals, 0)
                                    g_max = np.percentile(matrix_vals, 99.6)
                                else:
                                    g_min = np.percentile(matrix_vals, 0)
                                    g_max = np.percentile(matrix_vals, 99.95)
                            else:
                                g_min, g_max = seq_gt.min(), seq_gt.max()
                            
                            if g_max <= g_min: g_max = g_min + 1e-5
                            
                            channel_limits[ch] = (g_min, g_max)

                        # --- 2. 绘图与保存 ---
                        for ch in range(C):
                            # 确定通道名
                            if C > 1:
                                ch_name = "peeq" if ch == 0 else ("mises" if ch == 1 else f"ch{ch}")
                            else:
                                ch_name = self.config.get('data_type', 'peeq').lower()
                            
                            # 创建文件夹 (增加灰度图文件夹)
                            combined_save_path = os.path.join(current_sample_root, ch_name)
                            
                            gt_save_path = os.path.join(current_sample_root, f"{ch_name}_gt")
                            pred_save_path = os.path.join(current_sample_root, f"{ch_name}_pred")
                            error_save_path = os.path.join(current_sample_root, f"{ch_name}_error")
                            
                            gt_gray_save_path = os.path.join(current_sample_root, f"{ch_name}_gt_gray")
                            pred_gray_save_path = os.path.join(current_sample_root, f"{ch_name}_pred_gray")
                            error_gray_save_path = os.path.join(current_sample_root, f"{ch_name}_error_gray")
                            
                            os.makedirs(combined_save_path, exist_ok=True)
                            os.makedirs(gt_save_path, exist_ok=True)
                            os.makedirs(pred_save_path, exist_ok=True)
                            os.makedirs(error_save_path, exist_ok=True)
                            
                            os.makedirs(gt_gray_save_path, exist_ok=True)
                            os.makedirs(pred_gray_save_path, exist_ok=True)
                            os.makedirs(error_gray_save_path, exist_ok=True)
                            
                            vmin, vmax = channel_limits[ch]

                            # --- [A] 保存独立 Legend ---
                            legend_path = os.path.join(current_sample_root, f"{ch_name}_legend.png")
                            if not os.path.exists(legend_path): 
                                fig_leg = plt.figure(figsize=(1, 4))
                                ax_leg = fig_leg.add_axes([0.1, 0.05, 0.3, 0.9])
                                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                                cb = mpl.colorbar.ColorbarBase(ax_leg, cmap=plt.get_cmap("jet"),
                                                            norm=norm, orientation='vertical')
                                cb.ax.tick_params(labelsize=10)
                                plt.savefig(legend_path, bbox_inches='tight', transparent=True)
                                plt.close(fig_leg)

                            # --- [B] 定义保存图像的辅助函数 ---
                            def save_clean_image(img_data, save_path, v_min, v_max, mask):
                                # 原有逻辑：带 jet colormap 的伪彩图
                                norm_data = (img_data - v_min) / (v_max - v_min + 1e-9)
                                norm_data = np.clip(norm_data, 0, 1)
                                
                                colormap = plt.get_cmap("jet")
                                colored = colormap(norm_data) # RGBA
                                
                                if mask is not None:
                                    colored[mask] = [1.0, 1.0, 1.0, 1.0] 
                                
                                img_uint8 = (colored * 255).astype(np.uint8)
                                Image.fromarray(img_uint8).save(save_path)
                                
                            def save_gray_image(img_data, save_path, v_min, v_max, mask):
                                # 新增逻辑：不带 colormap 的数值灰度图，mask 部分置纯白
                                norm_data = (img_data - v_min) / (v_max - v_min + 1e-9)
                                norm_data = np.clip(norm_data, 0, 1)
                                
                                img_uint8 = (norm_data * 255).astype(np.uint8)
                                
                                if mask is not None:
                                    img_uint8[mask] = 255 # 将掩码部分（颗粒）置为 255 (纯白)
                                    
                                # 显式使用 'L' 模式保存为 8 位灰度图
                                Image.fromarray(img_uint8, mode='L').save(save_path)

                            # --- [C] 逐帧保存 ---
                            for t in range(T):
                                img_gt = gt_real[b, t, ch]
                                img_pred = pred_real[b, t, ch]
                                img_err = np.abs(img_gt - img_pred)
                                
                                file_name_stem = f"t_{t}"
                                if output_frame_names is not None:
                                    try:
                                        frame_path = Path(output_frame_names[b][t])
                                        base = frame_path.stem 
                                        if base.lower().startswith('peeq'): base = base[4:]
                                        elif base.lower().startswith('mises'): base = base[5:]
                                        file_name_stem = base
                                    except: pass
                                
                                # 1. 保存 RGB 伪彩图
                                save_clean_image(img_gt, 
                                                os.path.join(gt_save_path, f"{file_name_stem}.png"), 
                                                vmin, vmax, particle_mask_dilated)
                                
                                save_clean_image(img_pred, 
                                                os.path.join(pred_save_path, f"{file_name_stem}.png"), 
                                                vmin, vmax, particle_mask_dilated)

                                current_span = vmax - vmin
                                save_clean_image(img_err, 
                                                os.path.join(error_save_path, f"{file_name_stem}.png"), 
                                                0, current_span, particle_mask_dilated)
                                                
                                # 2. 保存 灰度 数值图 (新增部分)
                                save_gray_image(img_gt, 
                                                os.path.join(gt_gray_save_path, f"{file_name_stem}.png"), 
                                                vmin, vmax, particle_mask_dilated)
                                
                                save_gray_image(img_pred, 
                                                os.path.join(pred_gray_save_path, f"{file_name_stem}.png"), 
                                                vmin, vmax, particle_mask_dilated)
                                
                                save_gray_image(img_err, 
                                                os.path.join(error_gray_save_path, f"{file_name_stem}.png"), 
                                                0, current_span, particle_mask_dilated)

                            # --- [D] 组合图 (用于快速预览) ---
                            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                            if particle_mask_dilated is not None:
                                vis_gt = np.ma.masked_where(particle_mask_dilated, img_gt)
                                vis_pred = np.ma.masked_where(particle_mask_dilated, img_pred)
                                vis_err = np.ma.masked_where(particle_mask_dilated, img_err)
                            else:
                                vis_gt, vis_pred, vis_err = img_gt, img_pred, img_err

                            cmap = copy.copy(plt.get_cmap("jet"))
                            cmap.set_bad(color='white')
                            
                            # GT & Pred 使用 vmin/vmax
                            im1 = axes[0].imshow(vis_gt, cmap=cmap, vmin=vmin, vmax=vmax)
                            axes[0].set_title(f"GT")
                            axes[0].axis('off')
                            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                            im2 = axes[1].imshow(vis_pred, cmap=cmap, vmin=vmin, vmax=vmax)
                            axes[1].set_title(f"Pred")
                            axes[1].axis('off')
                            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                            
                            # Error 使用修正后的显示范围
                            im3 = axes[2].imshow(vis_err, cmap=cmap, vmin=0, vmax=current_span)
                            axes[2].set_title(f"Error (Shared Scale, Zero Base)")
                            axes[2].axis('off')
                            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(combined_save_path, f"{file_name_stem}.png"), dpi=100)
                            plt.close(fig)

            self.model.train()