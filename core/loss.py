import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import numpy as np
from typing import Optional

#loss.py에서는
#통계를 어떻게 모을지, 스케일을 어떻게 갱신할지, forward에서 현재 scale로 CAM을 정규화하고 align loss 계산


class PCCEVE8(nn.Module):
    """
    0 Anger
    1 Anticipation
    2 Disgust
    3 Fear
    4 Joy
    5 Sadness
    6 Surprise
    7 Trust
    Positive: Anticipation, Joy, Surprise, Trust
    Negative: Anger, Disgust, Fear, Sadness
    """

    def __init__(self, lambda_0=0):
        super(PCCEVE8, self).__init__()
        self.POSITIVE = {1, 4, 6, 7}
        self.NEGATIVE = {0, 2, 3, 5}

        self.lambda_0 = lambda_0

        self.f0 = nn.CrossEntropyLoss(reduce=False)

    def forward(self, y_pred: Tensor, y: Tensor):
        batch_size = y_pred.size(0)
        weight = [1] * batch_size

        out = self.f0(y_pred, y)
        _, y_pred_label = f.softmax(y_pred, dim=1).topk(k=1, dim=1)
        y_pred_label = y_pred_label.squeeze(dim=1)
        y_numpy = y.cpu().numpy()
        y_pred_label_numpy = y_pred_label.cpu().numpy()
        for i, y_numpy_i, y_pred_label_numpy_i in zip(range(batch_size), y_numpy, y_pred_label_numpy):
            if (y_numpy_i in self.POSITIVE and y_pred_label_numpy_i in self.NEGATIVE) or (
                    y_numpy_i in self.NEGATIVE and y_pred_label_numpy_i in self.POSITIVE):
                weight[i] += self.lambda_0
        weight_tensor = torch.from_numpy(np.array(weight)).cuda()
        out = out.mul(weight_tensor)
        out = torch.mean(out)

        return out

#추가한 함수, Intensity loss + CE loss

def _downsample_to(cam_map, sal_map, align_hw=56):
    # cam_map, sal_map: [B,S,1,H,W]
    B, S, _, Hc, Wc = cam_map.shape
    cam_ = cam_map.view(B*S, 1, Hc, Wc)
    sal_ = sal_map.view(B*S, 1, Hc, Wc)
    cam_ = F.interpolate(cam_, size=(align_hw, align_hw), mode="bilinear", align_corners=False)
    sal_ = F.interpolate(sal_, size=(align_hw, align_hw), mode="bilinear", align_corners=False)
    cam_map = cam_.view(B, S, 1, align_hw, align_hw)
    sal_map = sal_.view(B, S, 1, align_hw, align_hw)
    return cam_map, sal_map


class Intensity(nn.Module):
    """
    Align loss (지금은 L1) + (옵션) CAM calibration
    - cam_calib="epoch_p95": epoch 단위로 p95 통계를 모아서 scale 업데이트 (몇 epoch마다)
    """
    def __init__(
        self,
        eps: float = 1e-6,
        cam_calib: str = "epoch_p95",     # "none" | "epoch_p95"
        q: float = 0.95,                 # p95
        update_every_epochs: int = 1,    # 몇 epoch마다 scale 갱신할지
        clamp_max: Optional[float] = None,
    ):
        super().__init__()
        self.eps = eps
        self.cam_calib = cam_calib
        self.q = q
        self.update_every_epochs = max(1, int(update_every_epochs))
        self.clamp_max = clamp_max

        # 현재 적용 중인 scale(분모)
        self.register_buffer("scale", torch.tensor(1.0))

        # epoch 통계 누적용 (p95 합/개수)
        self.register_buffer("epoch_sum", torch.tensor(0.0))
        self.register_buffer("epoch_cnt", torch.tensor(0, dtype=torch.long))

        # 현재 epoch에서 통계를 모을지 여부 (파이썬 bool)
        self._collect_this_epoch = False

    def begin_epoch(self, epoch: int):
        """
        train loop에서 epoch 시작 시 호출.
        이번 epoch에서 scale을 갱신할 통계를 모을지 결정하고 누적값 초기화.
        """
        if self.cam_calib != "epoch_p95":
            self._collect_this_epoch = False
            return

        self._collect_this_epoch = (epoch % self.update_every_epochs == 0)
        # 누적 초기화
        self.epoch_sum.zero_()
        self.epoch_cnt.zero_()

    @torch.no_grad()
    def end_epoch(self):
        """
        train loop에서 epoch 끝날 때 호출.
        누적된 통계로 scale을 갱신.
        """
        if self.cam_calib != "epoch_p95":
            return
        if not self._collect_this_epoch:
            return

        if self.epoch_cnt.item() > 0:
            new_scale = (self.epoch_sum / self.epoch_cnt.float()).clamp_min(self.eps)
            self.scale.copy_(new_scale)

        self._collect_this_epoch = False

    @torch.no_grad()
    def _batch_p95(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*Seq, P] (P=spatial pixels)
        return: scalar p95 (batch 평균)
        """
        P = x.size(1)
        k = max(1, int((1.0 - self.q) * P))
        topk = torch.topk(x, k=k, dim=1, largest=True, sorted=False).values  # [N,k]
        p95 = topk.min(dim=1).values  # [N]
        return p95.mean()

    def _calibrate_cam(self, cam_map: torch.Tensor) -> torch.Tensor:
        """
        cam_map: [B,Seq,1,H,W] (또는 [B,Seq,H,W])
        - epoch_p95: epoch 중엔 scale 고정, 통계만 누적. epoch 끝에서 scale 업데이트.
        """
        if self.cam_calib == "none":
            return cam_map
        print("calib called | grad_enabled=", torch.is_grad_enabled(),
            "| collect=", self._collect_this_epoch,
            "| cnt=", int(self.epoch_cnt.item()),
            "| scale=", float(self.scale.item()))

        cam = cam_map.clamp_min(0)
        B, S = cam.size(0), cam.size(1)
        cam_flat = cam.view(B * S, -1)  # [B*Seq, H*W]

        # ✅ 통계 누적: train(grad enabled) + 이번 epoch이 수집 epoch일 때만
        if self.cam_calib == "epoch_p95" and torch.is_grad_enabled() and self._collect_this_epoch:
            with torch.no_grad():
                cur = self._batch_p95(cam_flat).clamp_min(self.eps)
                self.epoch_sum.add_(cur)
                self.epoch_cnt.add_(1)
                if self.epoch_cnt.item() == 1:
                    print("[stats] first update, cur=", float(cur.item()))

        # ✅ 정규화는 “현재 scale”로만 (epoch 중에는 고정)
        cam = cam / (self.scale + self.eps)

        if self.clamp_max is not None:
            cam = cam.clamp(0.0, self.clamp_max)

        return cam

    def forward(self, cam_map, sal_map):
        # ---- shape 통일 ----
        if cam_map.dim() == 4:
            cam_map = cam_map.unsqueeze(2)
            print("cam dim 4")
        if sal_map.dim() == 4:
            sal_map = sal_map.unsqueeze(2)
            print("sal dim 4")
        if sal_map.dim() == 6:
            sal_map = sal_map.mean(dim=3)
            print("sal dim 6")
        
        print("cam shape(loss func): ", cam_map.shape)

        # ---- 해상도 맞추기: saliency -> cam 해상도 ----
        B, S, _, Hc, Wc = cam_map.shape
        _, _, _, Hs, Ws = sal_map.shape
        if (Hs, Ws) != (Hc, Wc): #CAM과 Saliency의 H, W 비교
            sal_ = sal_map.view(B * S, 1, Hs, Ws) #view(): 텐서의 모양(shape)만 바꾸는 함수. interpolate()가 이 형식으로 받는 것을 기대함.
            sal_ = F.interpolate(sal_, size=(Hc, Wc), mode="bilinear", align_corners=False)  # 텐서를 원하는 크기로 리사이즈(업, 다운샘플링)하는 함수. saliency처럼 연속적인 값에는 보통 bilinear가 잘 맞는다.
            sal_map = sal_.view(B, S, 1, Hc, Wc) #다시 원래 모양으로 돌림

        # ---- CAM calibration (epoch 통계 기반) ----
        cam_map = self._calibrate_cam(cam_map) #CAM을 현재 scale(분모)로 나눠서 단위를 맞추는 것.

        # ---- align loss (RMSEL) ----
        cam = cam_map.clamp_min(self.eps)
        sal = sal_map.clamp_min(self.eps)
        return (torch.log(cam) - torch.log(sal)).abs().mean() #RMSEL 식


class IntensityGrad(Intensity):
    """
    Gradient-only align loss (Sobel 기반)
    - CAM calibration/resize 로직은 Intensity 그대로 사용
    - align은 |∂cam/∂x-∂sal/∂x| + |∂cam/∂y-∂sal/∂y| 만 사용
    """
    def __init__(self, *args, sobel_norm: float = 1.0/8.0, **kwargs):
        super().__init__(*args, **kwargs)

        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)
        self.sobel_norm = sobel_norm

    def _sobel(self, x_5d: torch.Tensor):
        """
        x_5d: [B, S, 1, H, W]
        return: gx, gy each [B, S, 1, H, W]
        """
        B, S, _, H, W = x_5d.shape
        x = x_5d.view(B * S, 1, H, W)
        gx = F.conv2d(x, self.sobel_x, padding=1) * self.sobel_norm
        gy = F.conv2d(x, self.sobel_y, padding=1) * self.sobel_norm
        gx = gx.view(B, S, 1, H, W)
        gy = gy.view(B, S, 1, H, W)
        return gx, gy

    def forward(self, cam_map, sal_map):
        # ---- shape 통일 (기존 Intensity와 동일) ----
        if cam_map.dim() == 4:  # [B,S,H,W]
            cam_map = cam_map.unsqueeze(2)
        if sal_map.dim() == 4:
            sal_map = sal_map.unsqueeze(2)
        if sal_map.dim() == 6:  # [B,S,1,D,H,W] -> snippet time mean
            sal_map = sal_map.mean(dim=3)

        # ---- 해상도 맞추기: saliency -> cam ----
        B, S, _, Hc, Wc = cam_map.shape
        _, _, _, Hs, Ws = sal_map.shape
        if (Hs, Ws) != (Hc, Wc):
            sal_ = sal_map.view(B * S, 1, Hs, Ws)
            sal_ = F.interpolate(sal_, size=(Hc, Wc), mode="bilinear", align_corners=False)
            sal_map = sal_.view(B, S, 1, Hc, Wc)

        # ---- CAM calibration (epoch_p95 등 기존 그대로) ----
        # cam_map, sal_map = _downsample_to(cam_map, sal_map, align_hw=56)
        cam_map = self._calibrate_cam(cam_map)

        # ---- Gradient loss only (Sobel) ----
        gx_c, gy_c = self._sobel(cam_map)
        gx_s, gy_s = self._sobel(sal_map)

        loss = (gx_c - gx_s).abs().mean() + (gy_c - gy_s).abs().mean()
        return loss

class IntensityNormal(Intensity):
    """
    Surface normal-only align loss
    - Intensity의 shape 정리/resize/cam calibration(epoch_p95 등) 로직 그대로 재사용
    - 2D map을 height field로 보고, (dx,dy)로 normal을 만든 뒤 cosine loss로 정렬
    """

    def __init__(self, *args,
                 diff: str = "central",   # "central" | "sobel"
                 z: float = 1.0,          # normal의 z 성분(기울기 민감도 조절)
                 eps_n: float = 1e-6,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.diff = diff
        self.z = z
        self.eps_n = eps_n

        if diff == "central":
            # 중앙차분(가볍고 스케일이 덜 큼)
            kx = torch.tensor([[0., 0., 0.],
                               [-1., 0., 1.],
                               [0., 0., 0.]], dtype=torch.float32).view(1, 1, 3, 3) * 0.5
            ky = torch.tensor([[0., -1., 0.],
                               [0.,  0., 0.],
                               [0.,  1., 0.]], dtype=torch.float32).view(1, 1, 3, 3) * 0.5
        elif diff == "sobel":
            # Sobel(엣지 민감, 출력이 더 큼)
            kx = torch.tensor([[-1., 0., 1.],
                               [-2., 0., 2.],
                               [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
            ky = torch.tensor([[-1., -2., -1.],
                               [ 0.,  0.,  0.],
                               [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
            # 필요하면 여기서 /8 같은 정규화도 가능 (원하면 추가)
        else:
            raise ValueError(f"Unknown diff={diff}")

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _grads(self, x_5d: torch.Tensor):
        """
        x_5d: [B,S,1,H,W]
        return: dx, dy each [B,S,1,H,W]
        """
        B, S, _, H, W = x_5d.shape
        x = x_5d.view(B * S, 1, H, W)
        dx = F.conv2d(x, self.kx, padding=1)
        dy = F.conv2d(x, self.ky, padding=1)
        dx = dx.view(B, S, 1, H, W)
        dy = dy.view(B, S, 1, H, W)
        return dx, dy

    def _normals(self, x_5d: torch.Tensor):
        """
        x_5d: [B,S,1,H,W]
        return: n [B,S,3,H,W] (unit normal)
        """
        dx, dy = self._grads(x_5d)
        B, S, _, H, W = x_5d.shape
        nz = torch.ones((B, S, 1, H, W), device=x_5d.device, dtype=x_5d.dtype)

        n = torch.cat([-dx, -dy, nz], dim=2)  # [B,S,3,H,W]
        n = n / (torch.linalg.norm(n, dim=2, keepdim=True) + self.eps_n)
        return n

    def forward(self, cam_map, sal_map):
        # ---- shape 통일 ----
        if cam_map.dim() == 4:  # [B,S,H,W]
            cam_map = cam_map.unsqueeze(2)
        if sal_map.dim() == 4:
            sal_map = sal_map.unsqueeze(2)
        if sal_map.dim() == 6:  # [B,S,1,D,H,W] -> snippet 내부 평균
            sal_map = sal_map.mean(dim=3)

        # ---- 해상도 맞추기: saliency -> cam ----
        B, S, _, Hc, Wc = cam_map.shape
        _, _, _, Hs, Ws = sal_map.shape
        if (Hs, Ws) != (Hc, Wc):
            sal_ = sal_map.view(B * S, 1, Hs, Ws)
            sal_ = F.interpolate(sal_, size=(Hc, Wc), mode="bilinear", align_corners=False)
            sal_map = sal_.view(B, S, 1, Hc, Wc)

        # ---- CAM calibration(너의 epoch_p95 등) ----
        # cam_map, sal_map = _downsample_to(cam_map, sal_map, align_hw=56)
        cam_map = self._calibrate_cam(cam_map)

        # ✅ 여기(= n_cam 만들기 직전)에 넣기
        dx_c, dy_c = self._grads(cam_map)
        dx_s, dy_s = self._grads(sal_map)
        print("[∇] mean|dx_cam|", dx_c.abs().mean().item(),
            "mean|dy_cam|", dy_c.abs().mean().item(),
            "mean|dx_sal|", dx_s.abs().mean().item(),
            "mean|dy_sal|", dy_s.abs().mean().item())

        # ---- surface normal loss ----
        n_cam = self._normals(cam_map)  # [B,S,3,H,W]
        n_sal = self._normals(sal_map)

        cos = (n_cam * n_sal).sum(dim=2).clamp(-1.0, 1.0)  # [B,S,H,W]
        loss = (1.0 - cos).mean()
        return loss
    
class Intensity_CE(nn.Module):
    def __init__(self, cls_loss, intensity_loss, lambda_intensity: float):
        super().__init__()
        self.cls_loss = cls_loss
        self.intensity_loss = intensity_loss
        self.lambda_intensity = lambda_intensity

    def forward(self, y_pred, y, cam_map=None, saliency_map=None):
        cls = self.cls_loss(y_pred, y)

        # ce_intensity 모드에서 cam_map이 없으면 “조용히” 넘어가면 디버깅이 지옥이라,
        # 강하게 에러 내는 걸 추천
        if self.lambda_intensity > 0: #lambda가 0 이상일 때만 호출.
            if cam_map is None or saliency_map is None:
                raise RuntimeError("ce_intensity 모드인데 cam_map/saliency_map이 전달되지 않았습니다.")
            print("cam shape(loss func): ", cam_map.shape)
            align = self.intensity_loss(cam_map, saliency_map)
            return cls + self.lambda_intensity * align
        print("***********************************")

        return cls

class IntensityCombo(nn.Module):
    """
    Combine existing align losses (e.g., RMSEL + Normal) without rewriting them.
    - Uses existing Intensity / IntensityNormal modules as-is
    - Stores last_terms for logging
    """
    def __init__(self, rmse_loss: nn.Module, normal_loss: nn.Module,
                 w_rmse: float = 1.0, w_normal: float = 1.0):
        super().__init__()
        self.rmse_loss = rmse_loss
        self.normal_loss = normal_loss
        self.w_rmse = w_rmse
        self.w_normal = w_normal
        self.last_terms = {}

    def begin_epoch(self, epoch: int):
        # epoch_p95 통계 수집이 둘 다 켜져있으면 둘 다 begin_epoch 호출
        if hasattr(self.rmse_loss, "begin_epoch"):
            self.rmse_loss.begin_epoch(epoch)
        if hasattr(self.normal_loss, "begin_epoch"):
            self.normal_loss.begin_epoch(epoch)

    @torch.no_grad()
    def end_epoch(self):
        if hasattr(self.rmse_loss, "end_epoch"):
            self.rmse_loss.end_epoch()
        if hasattr(self.normal_loss, "end_epoch"):
            self.normal_loss.end_epoch()

    def forward(self, cam_map, sal_map):
        rmse = self.rmse_loss(cam_map, sal_map)
        normal = self.normal_loss(cam_map, sal_map)
        total = self.w_rmse * rmse + self.w_normal * normal

        self.last_terms = {
            "rmse": float(rmse.detach().item()),
            "normal": float(normal.detach().item()),
            "total": float(total.detach().item()),
        }
        return total

# loss.py 안에 추가 (Intensity/IntensityGrad/IntensityNormal 아래쪽에 두면 됨)

class IntensityAll(IntensityNormal):
    """
    RMSEL + Grad + Normal을 한 번에 계산하는 align loss
    - preprocess/resize/downsample/calibration을 1회만 수행
    - last_terms에 각 항을 저장해서 로깅 가능
    """
    def __init__(
        self,
        *args,
        w_rmse: float = 1.0,
        w_grad: float = 1.0,
        w_normal: float = 1.0,
        align_hw: int = 56,              # 세 항 동일 해상도 권장
        sobel_norm: float = 1.0/8.0,     # grad 항 스케일
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.w_rmse = w_rmse
        self.w_grad = w_grad
        self.w_normal = w_normal
        self.align_hw = align_hw

        # Sobel kernel (Grad loss용)
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)
        self.sobel_norm = sobel_norm

        self.last_terms = {}

    def _sobel(self, x_5d: torch.Tensor):
        B, S, _, H, W = x_5d.shape
        x = x_5d.view(B * S, 1, H, W)
        gx = F.conv2d(x, self.sobel_x, padding=1) * self.sobel_norm
        gy = F.conv2d(x, self.sobel_y, padding=1) * self.sobel_norm
        return gx.view(B, S, 1, H, W), gy.view(B, S, 1, H, W)

    def forward(self, cam_map, sal_map):
        # ---- shape 통일 ----
        if cam_map.dim() == 4:  # [B,S,H,W]
            cam_map = cam_map.unsqueeze(2)
        if sal_map.dim() == 4:
            sal_map = sal_map.unsqueeze(2)
        if sal_map.dim() == 6:  # [B,S,1,D,H,W] -> snippet mean
            sal_map = sal_map.mean(dim=3)

        # ---- 해상도 맞추기: saliency -> cam ----
        B, S, _, Hc, Wc = cam_map.shape
        _, _, _, Hs, Ws = sal_map.shape
        if (Hs, Ws) != (Hc, Wc):
            sal_ = sal_map.view(B * S, 1, Hs, Ws)
            sal_ = F.interpolate(sal_, size=(Hc, Wc), mode="bilinear", align_corners=False)
            sal_map = sal_.view(B, S, 1, Hc, Wc)

        # ---- (선택) 공통 downsample ----
        if self.align_hw is not None and (Hc != self.align_hw or Wc != self.align_hw):
            cam_map, sal_map = _downsample_to(cam_map, sal_map, align_hw=self.align_hw)

        # ---- CAM calibration (epoch_p95 등) 1회만 ----
        cam_map = self._calibrate_cam(cam_map)

        # ---- RMSEL ----
        cam = cam_map.clamp_min(self.eps)
        sal = sal_map.clamp_min(self.eps)
        rmse = (torch.log(cam) - torch.log(sal)).abs().mean()

        # ---- Grad ----
        gx_c, gy_c = self._sobel(cam_map)
        gx_s, gy_s = self._sobel(sal_map)
        grad = (gx_c - gx_s).abs().mean() + (gy_c - gy_s).abs().mean()

        # ---- Normal ----
        # z가 실제로 영향 주도록: (기존 IntensityNormal은 z가 안 쓰이는 상태였음)
        dx_c, dy_c = self._grads(cam_map)
        dx_s, dy_s = self._grads(sal_map)
        nz = self.z * torch.ones((B, S, 1, cam_map.size(-2), cam_map.size(-1)),
                                 device=cam_map.device, dtype=cam_map.dtype)
        n_cam = torch.cat([-dx_c, -dy_c, nz], dim=2)
        n_sal = torch.cat([-dx_s, -dy_s, nz], dim=2)
        n_cam = n_cam / (torch.linalg.norm(n_cam, dim=2, keepdim=True) + self.eps_n)
        n_sal = n_sal / (torch.linalg.norm(n_sal, dim=2, keepdim=True) + self.eps_n)
        cos = (n_cam * n_sal).sum(dim=2).clamp(-1.0, 1.0)
        normal = (1.0 - cos).mean()

        total = self.w_rmse * rmse + self.w_grad * grad + self.w_normal * normal

        # 로깅용 저장
        self.last_terms = {
            "rmse": float(rmse.detach().item()),
            "grad": float(grad.detach().item()),
            "normal": float(normal.detach().item()),
            "total": float(total.detach().item()),
        }
        return total


def get_loss(opt):
    if opt.loss_func == 'ce':
        return nn.CrossEntropyLoss()
    elif opt.loss_func == 'pcce_ve8':
        return PCCEVE8(lambda_0=opt.lambda_0)
    elif opt.loss_func == 'ce_intensity':
        print("ce_intensity")
        cls = nn.CrossEntropyLoss()
        intensity = Intensity(
            cam_calib="epoch_p95",
            q=getattr(opt, "cam_q", 0.95),
            update_every_epochs=getattr(opt, "cam_update_every_epochs", 1),
            clamp_max=getattr(opt, "cam_clamp_max", None),
        )
        return Intensity_CE(cls, intensity, lambda_intensity=getattr(opt, "lambda_intensity", 1.0)) #우선 intensity의 영향을 ce와 동일하게 설정
    elif opt.loss_func == 'ce_intensity_grad':
        print("ce_intensity_grad")
        cls = nn.CrossEntropyLoss()
        intensity = IntensityGrad(
            cam_calib="epoch_p95",
            q=getattr(opt, "cam_q", 0.95),
            update_every_epochs=getattr(opt, "cam_update_every_epochs", 1),
            clamp_max=getattr(opt, "cam_clamp_max", None),
        )
        return Intensity_CE(cls, intensity, lambda_intensity=getattr(opt, "lambda_intensity", 1.0))
    elif opt.loss_func == "ce_intensity_normal":
        print("ce_intensity_normal")
        cls = nn.CrossEntropyLoss()
        intensity = IntensityNormal(
            cam_calib="epoch_p95",
            q=getattr(opt, "cam_q", 0.95),
            update_every_epochs=getattr(opt, "cam_update_every_epochs", 1),
            clamp_max=getattr(opt, "cam_clamp_max", None),
            diff=getattr(opt, "normal_diff", "central"),  # central / sobel
            z=getattr(opt, "normal_z", 1.0),
        )
        return Intensity_CE(cls, intensity, lambda_intensity=getattr(opt, "lambda_intensity", 1.0))
    elif opt.loss_func == "ce_intensity_rmse_normal":
        cls = nn.CrossEntropyLoss()

        rmse = Intensity(
            cam_calib="epoch_p95",
            q=getattr(opt, "cam_q", 0.95),
            update_every_epochs=getattr(opt, "cam_update_every_epochs", 1),
            clamp_max=getattr(opt, "cam_clamp_max", None),
        )

        normal = IntensityNormal(
            cam_calib=getattr(opt, "normal_cam_calib", "none"),  # 기본 none 추천
            q=getattr(opt, "cam_q", 0.95),
            update_every_epochs=getattr(opt, "cam_update_every_epochs", 1),
            clamp_max=getattr(opt, "cam_clamp_max", None),
            diff=getattr(opt, "normal_diff", "central"),
            z=getattr(opt, "normal_z", 1.0),
        )

        combo = IntensityCombo(
            rmse_loss=rmse,
            normal_loss=normal,
            w_rmse=getattr(opt, "w_rmse", 1.0),
            w_normal=getattr(opt, "w_normal", 1.0),
        )

        return Intensity_CE(cls, combo, lambda_intensity=getattr(opt, "lambda_intensity", 1.0))
    elif opt.loss_func == "ce_intensity_all":
        cls = nn.CrossEntropyLoss()
        intensity = IntensityAll(
            cam_calib="epoch_p95",
            q=getattr(opt, "cam_q", 0.95),
            update_every_epochs=getattr(opt, "cam_update_every_epochs", 1),
            clamp_max=getattr(opt, "cam_clamp_max", None),
            diff=getattr(opt, "normal_diff", "central"),
            z=getattr(opt, "normal_z", 1.0),
            align_hw=getattr(opt, "align_hw", None),
            w_rmse=getattr(opt, "w_rmse", 1.0),
            w_grad=getattr(opt, "w_grad", 0.0),
            w_normal=getattr(opt, "w_normal", 1.0),
        )
        return Intensity_CE(cls, intensity, lambda_intensity=getattr(opt, "lambda_intensity", 1.0))
    else:
        raise Exception
