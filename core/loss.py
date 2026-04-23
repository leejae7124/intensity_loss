import torch.nn as nn
import torch.nn.functional as f
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

        # ---- 해상도 맞추기: saliency -> cam 해상도 ----
        B, S, _, Hc, Wc = cam_map.shape
        _, _, _, Hs, Ws = sal_map.shape
        if (Hs, Ws) != (Hc, Wc): #CAM과 Saliency의 H, W 비교
            sal_ = sal_map.view(B * S, 1, Hs, Ws) #view(): 텐서의 모양(shape)만 바꾸는 함수. interpolate()가 이 형식으로 받는 것을 기대함.
            sal_ = f.interpolate(sal_, size=(Hc, Wc), mode="bilinear", align_corners=False)  # 텐서를 원하는 크기로 리사이즈(업, 다운샘플링)하는 함수. saliency처럼 연속적인 값에는 보통 bilinear가 잘 맞는다.
            sal_map = sal_.view(B, S, 1, Hc, Wc) #다시 원래 모양으로 돌림

        # ---- CAM calibration (epoch 통계 기반) ----
        cam_map = self._calibrate_cam(cam_map) #CAM을 현재 scale(분모)로 나눠서 단위를 맞추는 것.

        # ---- align loss (RMSEL) ----
        cam = cam_map.clamp_min(self.eps)
        sal = sal_map.clamp_min(self.eps)
        return (torch.log(cam) - torch.log(sal)).abs().mean() #RMSEL 식


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
            align = self.intensity_loss(cam_map, saliency_map)
            return cls + self.lambda_intensity * align

        return cls


def get_loss(opt):
    if opt.loss_func == 'ce':
        return nn.CrossEntropyLoss()
    elif opt.loss_func == 'pcce_ve8':
        return PCCEVE8(lambda_0=opt.lambda_0)
    elif opt.loss_func == 'ce_intensity':
        cls = nn.CrossEntropyLoss()
        intensity = Intensity(
            cam_calib="epoch_p95",
            q=getattr(opt, "cam_q", 0.95),
            update_every_epochs=getattr(opt, "cam_update_every_epochs", 1),
            clamp_max=getattr(opt, "cam_clamp_max", None),
        )
        return Intensity_CE(cls, intensity, lambda_intensity=getattr(opt, "lambda_intensity", 1.0)) #우선 intensity의 영향을 ce와 동일하게 설정
    else:
        raise Exception
