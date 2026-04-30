from core.utils import AverageMeter, process_data_item, run_model_loss, calculate_accuracy

import time
import torch
import torchvision

def _unwrap(model):
    return model.module if hasattr(model, "module") else model

def _pick_two_params(named_params):
    named_params = list(named_params)
    if len(named_params) == 0:
        return []
    if len(named_params) == 1:
        return [named_params[0]]
    return [named_params[0], named_params[-1]]  # 첫/마지막 파라미터만 체크

@torch.no_grad()
def snapshot_backbone_params(model):
    m = _unwrap(model)
    snap = {}
    for name, p in _pick_two_params(m.resnet.named_parameters()):
        snap[name] = p.detach().clone()
    return snap

def print_backbone_grad(model, tag=""):
    m = _unwrap(model)
    pairs = _pick_two_params(m.resnet.named_parameters())
    if not pairs:
        print(f"[{tag}] resnet has no parameters?")
        return

    print(f"[{tag}] --- ResNet grad check ---")
    for name, p in pairs:
        rg = p.requires_grad
        gnone = (p.grad is None)
        gnorm = None if gnone else float(p.grad.detach().norm().item())
        print(f"[{tag}] {name:30s} requires_grad={rg} grad_none={gnone} grad_norm={gnorm}")

@torch.no_grad()
def print_backbone_update(model, snap, tag=""):
    m = _unwrap(model)
    pairs = _pick_two_params(m.resnet.named_parameters())
    print(f"[{tag}] --- ResNet weight delta check ---")
    for name, p in pairs:
        if name not in snap:
            continue
        delta = (p.detach() - snap[name]).abs().max().item()
        print(f"[{tag}] {name:30s} max|Δw|={delta:.6e}")

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, class_names, writer):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))

    # epoch 시작 직후, epoch 단위 CAM 통계를 쓰기 위한 작업.
    if opt.loss_func.startswith("ce_intensity"):
        criterion.intensity_loss.begin_epoch(epoch)

        

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    for i, data_item in enumerate(data_loader): #data_loader는 main.py에서 생성되어 train_epoch 함수로 전달된다.
        # print("*****", data_item[3])
        # print("시작함????")
        visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item) #visual, saliency, target, audio, visualization_item, batch
        # if torch.isnan(saliency_map).any():
        #     print("NaN detected in saliency_map")
        # if torch.isinf(saliency_map).any():
        #     print("Inf detected in saliency_map")
        # print("saliency_map stats:", saliency_map.min(), saliency_map.max())
        # print("📦 [Train] saliency_map shape:", saliency_map.shape)
        video_id = visualization_item[0]
        # print("visualization: ", visualization_item)
        # print("video_id: ", video_id)
        data_time.update(time.time() - end_time)

        # warmup_epoch는 1로 둔다고 가정 (epoch 번호가 1부터 시작하므로 epoch==1이 첫 epoch)
        #warmup = (opt.loss_func == "ce_intensity" and epoch == 1)
        warmup = (opt.loss_func.startswith("ce_intensity") and epoch == 1) #cam 통계만 쌓고, 업데이트는 ce로만 하자.
        if warmup:
            # 1) CAM 생성 (grad enabled 상태, model.train 상태)
            y_pred, alpha, beta, gamma, cam_map = model(
                visual, audio, target_class=target, compute_gradcam=True
            )

            # 2) ✅ 통계만 누적 (detach로 그래프 부담 줄임)
            _ = criterion.intensity_loss._calibrate_cam(cam_map.detach())

            # 3) ✅ 업데이트는 CE만
            output = y_pred
            loss = criterion.cls_loss(y_pred, target)  # Intensity_CE 내부 CE 모듈 사용
        else:
            output, loss = run_model_loss(opt, [visual, target, audio, saliency_map],
                                        model, criterion, i, print_attention=False)

        # run_model()으로부터 loss를 받아서 backprop -> run_model()에 alignment loss 작성 필요해보임
        # output, loss = run_model_loss(opt, [visual, target, audio, saliency_map], model, criterion, i, print_attention=False)

        iter = (epoch - 1) * len(data_loader) + (i + 1)   # ✅ iter 먼저 정의

        m = model.module if hasattr(model, "module") else model

        if hasattr(m, "last_cam_map") and (i % 200 == 0):
            cam_vis = m.last_cam_map[0]           # [Seq,H,W]  (배치 0)
            grid = torchvision.utils.make_grid(
                cam_vis.unsqueeze(1),             # [Seq,1,H,W]
                nrow=cam_vis.size(0),
                normalize=True
            )
            writer.add_image("cam/sample0_seq", grid, iter)

        # (A) gate 파라미터(학습되는 값)
        # writer.add_scalar("gate/param_b", m.gate_b.item(), iter)
        # writer.add_scalar("gate/param_c", m.gate_c.item(), iter)
        # writer.add_scalar("gate/param_d", m.gate_d.item(), iter)
        # writer.add_scalar("gate/param_temp", m.gate_temp.item(), iter)

        # # (B) gate 값 통계(이미 buffer로 저장됨)
        # writer.add_scalar("gate/gate_mean", m.last_gate_mean.item(), iter)
        # writer.add_scalar("gate/gate_min",  m.last_gate_min.item(), iter)
        # writer.add_scalar("gate/gate_max",  m.last_gate_max.item(), iter)

        # # (C) 기여도(각 항) - 일단 mean만
        # if m.last_term_overlap is not None:
        #     writer.add_scalar("gate/term_overlap_mean", m.last_term_overlap.mean().item(), iter)
        #     writer.add_scalar("gate/term_sal_mean",     m.last_term_sal.mean().item(), iter)
        #     writer.add_scalar("gate/term_cam_mean",     m.last_term_cam.mean().item(), iter)


        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)

        # Backward and optimize
        optimizer.zero_grad()

        # (2) step 전에 weight 스냅샷(업데이트 확인용)
        snap = snapshot_backbone_params(model)   # ✅ optimizer.step 전

        loss.backward()
        if i % 50 == 0:  # 너무 자주 찍히면 시끄러우니 주기 조절
            print_backbone_grad(model, tag=f"iter{i}")
        
        optimizer.step()

        # (5) step 후 weight 변화 확인
        if i % 50 == 0:
            print_backbone_update(model, snap, tag=f"iter{i}")

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        iter = (epoch - 1) * len(data_loader) + (i + 1)
        writer.add_scalar('train/batch/loss', losses.val, iter)
        writer.add_scalar('train/batch/acc', accuracies.val, iter)

        if opt.debug:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=accuracies))
        
    # epoch 끝
    if opt.loss_func.startswith("ce_intensity"):
        criterion.intensity_loss.end_epoch()

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))
    print("Train loss: {:.4f}".format(losses.avg))
    print("Train acc: {:.4f}".format(accuracies.avg))

    writer.add_scalar('train/epoch/loss', losses.avg, epoch)
    writer.add_scalar('train/epoch/acc', accuracies.avg, epoch)
