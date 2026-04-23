import torch
from torch.cuda import device_count

from opts_tsl_school import parse_opts
from core.model import generate_model_intensity
from core.loss import get_loss
from core.utils import local2global_path, get_spatial_transform, get_saliency_transform
from core.dataset2 import get_test_set, get_data_loader

from transforms.temporal import TSN
from transforms.target import ClassLabel

from test_tsl import test_epoch


def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)

    # 학습 때와 동일하게
    opt.saliency_level = 'feature_map'

    # test dataset
    spatial_transform = get_spatial_transform(opt, 'test')
    saliency_transform = get_saliency_transform(opt, 'test', spatial_transform)
    temporal_transform = TSN(
        seq_len=opt.seq_len,
        snippet_duration=opt.snippet_duration,
        center=True
    )
    target_transform = ClassLabel()

    test_data = get_test_set(
        opt,
        spatial_transform,
        temporal_transform,
        target_transform,
        saliency_transform
    )
    test_loader = get_data_loader(opt, test_data, shuffle=False)

    # model
    model, parameters = generate_model_intensity(opt)
    criterion = get_loss(opt).cuda()

    # checkpoint load
    ckpt = torch.load(opt.checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    # test
    acc, macro_f1, preds, targets, video_ids = test_epoch(test_loader, model, criterion, opt)
    print("Final Test Acc     :", acc)
    print("Final Test Macro F1:", macro_f1)


if __name__ == "__main__":
    main()