
#saliency 적용
import sys
import argparse
import torch

from core.model import generate_model_intensity
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform, get_saliency_transform
from core.dataset2 import get_training_set, get_validation_set, get_test_set, get_data_loader

from transforms.temporal import TSN
from transforms.target import ClassLabel

from train_loss import train_epoch
from validation_loss import val_epoch

from torch.utils.data import DataLoader
from torch.cuda import device_count

from tensorboardX import SummaryWriter

# def get_audio_stats(data_loader):
#     # 합계, 제곱의 합계, 총 개수를 저장할 변수
#     sum_val = 0.0
#     sum_sq_val = 0.0
#     count = 0
    
#     print("Calculating audio stats from training data...")
#     for data_item in data_loader:
#         audio_batch = data_item[3] # 데이터 로더에서 오디오 텐서 추출
        
#         # 현재 배치의 합계와 제곱의 합계를 누적
#         sum_val += torch.sum(audio_batch)
#         sum_sq_val += torch.sum(audio_batch.pow(2))
        
#         # 현재 배치의 총 원소 개수를 누적
#         count += audio_batch.numel()

#     # 최종 평균 계산
#     mean = sum_val / count
    
#     # 최종 표준편차 계산: Var(X) = E[X^2] - (E[X])^2
#     std = torch.sqrt(sum_sq_val / count - mean.pow(2))
#     print("mean: ", mean)
#     print("std: ", std)
    
#     return mean, std
def load_parse_opts():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env", choices=["lab", "school"], default="lab")

    args, remaining = bootstrap.parse_known_args()

    # 나머지 인자들은 실제 opts parser로 넘기기 위해 sys.argv 재구성
    sys.argv = [sys.argv[0]] + remaining

    if args.env == "school":
        print("school")
        from opts_tsl_school import parse_opts
    else:
        print("lab")
        from opts_tsl import parse_opts

    return parse_opts

def main():
    parse_opts = load_parse_opts()
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)

    # train
    spatial_transform = get_spatial_transform(opt, 'train') #여기에서 Preprocessing 객체 생성
    saliency_transform = get_saliency_transform(opt, 'train', spatial_transform)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform)
    train_loader = get_data_loader(opt, training_data, shuffle=True)

    # validation
    spatial_transform = get_spatial_transform(opt, 'val')
    saliency_transform = get_saliency_transform(opt, 'val', spatial_transform)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)

    opt.saliency_level = 'feature_map'
    # print(f"Calculated Audio Stats -> Mean: {opt.audio_mean:.4f}, Std: {opt.audio_std:.4f}")
    # --- [디버깅 1] generate_model 호출 직전 opt 값 확인 ---
    print("\n--- Values before calling generate_model ---")
    # print(f"opt.audio_mean: {opt.audio_mean}")
    # print(f"opt.audio_std: {opt.audio_std}\n")

    model, parameters = generate_model_intensity(opt)

    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    writer = SummaryWriter(logdir=opt.log_path)

    

    for i in range(1, opt.n_epochs + 1):
        train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names, writer)
        val_epoch(i, val_loader, model, criterion, opt, writer, optimizer)

    writer.close()


if __name__ == "__main__":
    main()

"""
python main.py --expr_name demo
"""