import os
import datetime
import shutil
import torch

from transforms.spatial import Preprocessing, Preprocessing_saliency


def local2global_path(opt):
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.audio_path = os.path.join(opt.root_path, opt.audio_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.saliency_path = os.path.join(opt.root_path, opt.saliency_path)
        if opt.debug:
            opt.result_path = "debug"
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.expr_name == '':
            now = datetime.datetime.now()
            now = now.strftime('result_%Y%m%d_%H%M%S')
            opt.result_path = os.path.join(opt.result_path, now)
        else:
            opt.result_path = os.path.join(opt.result_path, opt.expr_name)

            if os.path.exists(opt.result_path):
                shutil.rmtree(opt.result_path)
            os.mkdir(opt.result_path)

        opt.log_path = os.path.join(opt.result_path, "tensorboard")
        opt.ckpt_path = os.path.join(opt.result_path, "checkpoints")
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        if not os.path.exists(opt.ckpt_path):
            os.mkdir(opt.ckpt_path)
    else:
        raise Exception


# def get_spatial_transform(opt, mode): #증강 끄기
#     if mode == "train":
#         return Preprocessing(size=opt.sample_size, is_aug=False, center=False)
#     elif mode == "val":
#         return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
#     elif mode == "test":
#         return Preprocessing(size=opt.sample_size, is_aug=False, center=False)
#     else:
#         raise Exception

def get_spatial_transform(opt, mode): #증강 켜기
    if mode == "train":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=False)
    elif mode == "val":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    elif mode == "test":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    else:
        raise Exception

def get_saliency_transform(opt, mode, spatial_transform):
    if mode == "train":
        return Preprocessing_saliency(original_preprocessing_instance=spatial_transform)
    elif mode == "val":
        return Preprocessing_saliency(original_preprocessing_instance=spatial_transform)
    elif mode == "test":
        return Preprocessing_saliency(original_preprocessing_instance=spatial_transform)
    else:
        raise Exception


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


# def process_data_item(opt, data_item):
#     visual, saliency, target, audio, visualization_item = data_item #snippets, saliency_snippets, target, audios, visualization_item
#     target = target.cuda()
#     saliency = saliency.cuda()

#     visual = visual.cuda()
#     audio = audio.cuda()
#     assert visual.size(0) == audio.size(0) == saliency.size(0)
#     batch = visual.size(0)
#     return visual, saliency, target, audio, visualization_item, batch
# core/utils.py (process_data_item 함수 수정)

def process_data_item(opt, data_item):
    visual, saliency_data_tensor, target, audios_data_tensor, visualization_item_data_batch = data_item 

    # [디버깅 1] CUDA 전송 전 각 텐서의 타입, 형태, 값 범위 확인
    # print("\n--- Debugging Data Before CUDA Transfer ---")
    
    # Visual (비디오 프레임)
    # if not isinstance(visual, torch.Tensor): raise TypeError(f"Visual is not a Tensor: {type(visual)}")
    # print(f"Visual: shape={visual.shape}, dtype={visual.dtype}, min={visual.min().item()}, max={visual.max().item()}")
    # if torch.isnan(visual).any() or torch.isinf(visual).any(): raise ValueError("Visual contains NaN/Inf values.")

    # # Saliency (Saliency Map)
    # if not isinstance(saliency_data_tensor, torch.Tensor): raise TypeError(f"Saliency is not a Tensor: {type(saliency_data_tensor)}")
    # print(f"Saliency: shape={saliency_data_tensor.shape}, dtype={saliency_data_tensor.dtype}, min={saliency_data_tensor.min().item()}, max={saliency_data_tensor.max().item()}")
    # if torch.isnan(saliency_data_tensor).any() or torch.isinf(saliency_data_tensor).any(): raise ValueError("Saliency contains NaN/Inf values.")

    # # Target (레이블)
    # if not isinstance(target, torch.Tensor): raise TypeError(f"Target is not a Tensor: {type(target)}")
    # print(f"Target: shape={target.shape}, dtype={target.dtype}, values={target.tolist()}") # tolist()로 값 확인

    # # Audio (오디오 특징) - opt.need_audio에 따라 다르게 처리
    
    # if not isinstance(audios_data_tensor, torch.Tensor): raise TypeError(f"Audio is not a Tensor: {type(audios_data_tensor)}")
    # print(f"Audio: shape={audios_data_tensor.shape}, dtype={audios_data_tensor.dtype}, min={audios_data_tensor.min().item()}, max={audios_data_tensor.max().item()}")
    # if torch.isnan(audios_data_tensor).any() or torch.isinf(audios_data_tensor).any(): raise ValueError("Audio contains NaN/Inf values.")

    # print("--- End Debugging Data Before CUDA Transfer ---\n")

    # GPU로 데이터 이동
    visual = visual.cuda()
    saliency_data_tensor = saliency_data_tensor.cuda() 
    target = target.cuda()
    
    
    audios_data_tensor = audios_data_tensor.cuda()

    if not hasattr(process_data_item, "_printed_saliency_stats"):
        print("[saliency] shape:", tuple(saliency_data_tensor.shape),
              "dtype:", saliency_data_tensor.dtype,
              "min/max:", saliency_data_tensor.min().item(), saliency_data_tensor.max().item(),
              "mean:", saliency_data_tensor.mean().item())
        process_data_item._printed_saliency_stats = True


    # 배치 크기 assert는 그대로 유지
    
    assert visual.size(0) == audios_data_tensor.size(0) == saliency_data_tensor.size(0) == target.size(0), \
            "Batch sizes of visual, audio, saliency, or target do not match."
    
    batch = visual.size(0)
    
    return visual, saliency_data_tensor, target, audios_data_tensor, visualization_item_data_batch, batch

def run_model(opt, inputs, model, criterion, i=0, print_attention=True, period=30, return_attention=False):
    visual, target, audio , saliency_map= inputs

    # train 때만 gradcam 켜기
    compute_gradcam = True

    # outputs = model(visual, audio, saliency_map,  target_class=None,# (선택) GT로 CAM/gate 만들고 싶다면
    #                 compute_gradcam=compute_gradcam)
    outputs = model(visual, audio, saliency_map)
    y_pred, alpha, beta, gamma = outputs
    loss = criterion(y_pred, target)
    if i % period == 0 and print_attention:
        print('====alpha====')
        print(alpha[:, 0, :])
        print('====beta====')
        print(beta[:, 0, 0:512:32])
        print('====gamma====')
        print(gamma)
    if not return_attention:
        return y_pred, loss
    else:
        return y_pred, loss, [alpha, beta, gamma]


def run_model_loss(opt, inputs, model, criterion, i=0, print_attention=True, period=30, return_attention=False):
    visual, target, audio , saliency_map = inputs

    # ✅ Grad-CAM은 train + grad enabled 일 때만
    need_align = opt.loss_func.startswith("ce_intensity")  # ce_intensity, ce_intensity_grad, ...
    do_cam = need_align and model.training and torch.is_grad_enabled()

    if do_cam:
        y_pred, alpha, beta, gamma, cam_map = model(
            visual, audio, target_class=target, compute_gradcam=True
        )
        # 2) loss 항 분리 계산 (한 번만 계산)
        cls = criterion.cls_loss(y_pred, target)                      # CE
        align = criterion.intensity_loss(cam_map, saliency_map)       # RMSEL or Grad
        lam = float(getattr(criterion, "lambda_intensity", 1.0))
        loss = cls + lam * align
        # loss = criterion(y_pred, target, cam_map=cam_map, saliency_map=saliency_map)
    else:
        # val/test 또는 CE-only 상황
        y_pred, alpha, beta, gamma = model(visual, audio, compute_gradcam=False)

        if need_align:
            # ✅ ce_intensity 옵션이어도 val에서는 CE만 보겠다
            # Intensity_CE 안에 cls_loss가 있으니 그걸 사용
            loss = criterion.cls_loss(y_pred, target)
        else:
            loss = criterion(y_pred, target)
    
    # loss = criterion(y_pred, target)
    if i % period == 0 and print_attention:
        print('====alpha====')
        print(alpha[:, 0, :])
        print('====beta====')
        print(beta[:, 0, 0:512:32])
        print('====gamma====')
        print(gamma)
    if not return_attention:
        return y_pred, loss
    else:
        return y_pred, loss, [alpha, beta, gamma]

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    values, indices = outputs.topk(k=1, dim=1, largest=True)
    pred = indices
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elements = correct.float()
    n_correct_elements = n_correct_elements.sum()
    n_correct_elements = n_correct_elements.item()
    return n_correct_elements / batch_size