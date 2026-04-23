# test_tsl.py
import torch
import torch.nn as nn
from core.utils import AverageMeter, process_data_item, calculate_accuracy

@torch.no_grad()
def compute_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    """
    y_true, y_pred: shape [N], int64
    macro-F1 = (1/C) * Σ_c F1_c
    F1_c = 2*TP / (2*TP + FP + FN)
    """
    f1_sum = 0.0
    eps = 1e-12

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()

        denom = 2 * tp + fp + fn
        f1_c = (2 * tp) / (denom + eps) if denom > 0 else 0.0
        f1_sum += f1_c

    return f1_sum / max(num_classes, 1)


@torch.no_grad()
def test_epoch(data_loader, model, criterion, opt):
    print("# -------------------------------------------------- #")
    print("Test model")
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    all_preds = []
    all_targets = []
    all_video_ids = []

    # ✅ test는 CE-only로 보는 게 일반적
    ce_only = nn.CrossEntropyLoss().cuda()

    for i, data_item in enumerate(data_loader):
        visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)

        # ✅ intensity 방식: test에서는 cam/align 안 만들고 forward만
        out = model(visual, audio, compute_gradcam=False)  # forward가 compute_gradcam 인자 받는 경우
        # 모델이 (output, alpha, beta, gamma) 또는 (output, alpha, beta, gamma, cam_map)을 반환할 수 있으니 안전하게 처리
        if isinstance(out, (list, tuple)) and len(out) == 5:
            output, alpha, beta, gamma, _ = out
        else:
            output, alpha, beta, gamma = out

        loss = ce_only(output, target)

        acc = calculate_accuracy(output, target)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)

        preds = output.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())
        all_video_ids.extend(visualization_item)

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    num_classes = getattr(opt, "n_classes", output.size(1))
    macro_f1 = compute_macro_f1(y_true, y_pred, num_classes)

    print("Test loss    : {:.4f}".format(losses.avg))
    print("Test acc     : {:.4f}".format(accuracies.avg))
    print("Test macro F1: {:.4f}".format(macro_f1))

    return accuracies.avg, macro_f1, y_pred.tolist(), y_true.tolist(), all_video_ids