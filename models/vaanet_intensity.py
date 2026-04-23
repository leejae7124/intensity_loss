import torch
import torch.nn as nn
import torchvision
from models.visual_stream import VisualStream
#VAANet Dac(Saliency X) 코드에서 Intensity loss 프로세스에 맞게 수정 중. (Grad-CAM 구하는 과정이 포함되어 있어서 이 코드에서 수정하기 시작함)


class VAANet_intensity(VisualStream):
    def __init__(self,
                 snippet_duration=16,
                 sample_size=112,
                 n_classes=8,
                 seq_len=10,
                 pretrained_resnet101_path='',
                 audio_embed_size=256,
                 audio_n_segments=16,
                 audio_mean=0.0,
                 audio_std=1.0):
        super(VAANet_intensity, self).__init__(
            snippet_duration=snippet_duration,
            sample_size=sample_size,
            n_classes=n_classes,
            seq_len=seq_len,
            pretrained_resnet101_path=pretrained_resnet101_path
        )

        # self.register_buffer('audio_mean', torch.tensor(audio_mean))
        # self.register_buffer('audio_std', torch.tensor(audio_std))

        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        

        a_resnet = torchvision.models.resnet18(pretrained=True)
        a_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        a_avgpool = nn.AvgPool2d(kernel_size=[8, 2])
        a_modules = [a_conv1] + list(a_resnet.children())[1:-2] + [a_avgpool]
        self.a_resnet = nn.Sequential(*a_modules)
        self.a_fc = nn.Sequential(
            nn.Linear(a_resnet.fc.in_features, self.audio_embed_size),
            nn.BatchNorm1d(self.audio_embed_size),
            nn.Tanh()
        )

        self.aa_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.audio_embed_size, 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.audio_n_segments, self.audio_n_segments, bias=True),
            'relu': nn.ReLU(),
        })

        self.av_fc = nn.Linear(self.audio_embed_size + self.hp['k'], self.n_classes)

        self._gc_enable = False
        self._gc_activ = None

        # target = self._find_last_conv(self.resnet)
        target = self.conv0
        self._gc_handle = target.register_forward_hook(self._gc_save_activation)

    def _find_last_conv(self, module: nn.Module) -> nn.Module:
        last = None
        for m in module.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                last = m
        if last is None:
            raise RuntimeError("No Conv2d/Conv3d found in self.resnet for Grad-CAM target layer.")
        return last

    # def _gc_save_activation(self, module, input, output): #module: hook을 건 레이어, input: 그 레이어에 들어온 입력, out: 그 레이어의 출력
    #     if not self._gc_enable: #gc_enable이 false면 아무것도 안 하고 끝
    #         return
    #     # Grad-CAM용 activation을 "그래프에 붙는" 텐서로 교체
    #     out_det = output.detach()
    #     out_det.requires_grad_(True)

    #     self._gc_activ = out_det

    #     # forward hook은 return 값을 주면 "출력 교체"가 가능함
    #     return out_det
    def _gc_save_activation(self, module, input, output):
        if not self._gc_enable:
            return
        # conv0는 grad가 자연히 붙으므로 그대로 저장
        self._gc_activ = output
        # return None  (출력 교체 안 함)

    def forward(self, visual, audio, target_class=None, compute_gradcam=False):
        print("visual ", visual.shape)

        if not hasattr(self, "_printed_v1"):
            print("[VAANet] visual raw shape:", visual.shape)
            print("[VAANet] visual raw min/max:", visual.min().item(), visual.max().item())
            self._printed_v1 = True
        
        visual = visual.transpose(0, 1).contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)

        if not hasattr(self, "_printed_v2"):
            print("[VAANet] visual after norm min/max:", visual.min().item(), visual.max().item())
            self._printed_v2 = True

        # Visual branch
        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()
        
        self._gc_enable = compute_gradcam #grad-cam 계산할 건지
        self._gc_activ = None
        if compute_gradcam:
            F = self.resnet(visual)   # gradient 계산 허용, 계산을 허용한 것이지 resnet 파라미터를 업데이트 한다는 것은 아님.
            F = torch.squeeze(F, dim=2)
            F = torch.flatten(F, start_dim=2)
            print("mimimim***")
        else:
            with torch.no_grad(): #autograd 그래프 기록을 끄는 것.
                # ResNet을 통과시켜 피처맵 생성
                F = self.resnet(visual)
                F = torch.squeeze(F, dim=2)
                F = torch.flatten(F, start_dim=2)
                print("mimimimimi")
        F = self.conv0(F)  # [B x 512 x 16]

        Hs = self.sa_net['conv'](F)
        Hs = torch.squeeze(Hs, dim=1)
        Hs = self.sa_net['fc'](Hs)
        As = self.sa_net['softmax'](Hs)
        As = torch.mul(As, self.hp['m'])
        alpha = As.view(seq_len, batch, self.hp['m'])

        fS = torch.mul(F, torch.unsqueeze(As, dim=1).repeat(1, self.hp['k'], 1))

        G = fS.transpose(1, 2).contiguous()
        Hc = self.cwa_net['conv'](G)
        Hc = torch.squeeze(Hc, dim=1)
        Hc = self.cwa_net['fc'](Hc)
        Ac = self.cwa_net['softmax'](Hc)
        Ac = torch.mul(Ac, self.hp['k'])
        beta = Ac.view(seq_len, batch, self.hp['k'])

        fSC = torch.mul(fS, torch.unsqueeze(Ac, dim=2).repeat(1, 1, self.hp['m']))
        fSC = torch.mean(fSC, dim=2)
        fSC = fSC.view(seq_len, batch, self.hp['k']).contiguous()
        fSC = fSC.permute(1, 2, 0).contiguous()

        Ht = self.ta_net['conv'](fSC)
        Ht = torch.squeeze(Ht, dim=1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(batch, seq_len)

        fSCT = torch.mul(fSC, torch.unsqueeze(At, dim=1).repeat(1, self.hp['k'], 1))
        fSCT = torch.mean(fSCT, dim=2)  # [bs x 512]

        # Audio branch
        print("\n--- Inside forward pass ---")
        print(f"Before Normalization - Min: {torch.min(audio)}, Max: {torch.max(audio)}")


        # audio = (audio - self.audio_mean) / self.audio_std
        print("audio shape: ", audio.shape)

        print(f"After Normalization - Min: {torch.min(audio)}, Max: {torch.max(audio)}")
        bs = audio.size(0)
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(self.audio_n_segments, dim=0)
        # print("audio shape(2): ", audio.shape)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        audio = torch.unsqueeze(audio, dim=1)
        print("audio shape(3): ", audio.shape)
        audio = self.a_resnet(audio)
        audio = torch.flatten(audio, start_dim=1).contiguous()
        audio = self.a_fc(audio)
        audio = audio.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        audio = audio.permute(1, 2, 0).contiguous()

        print("audio shape(4): ", audio.shape)

        Ha = self.aa_net['conv'](audio)
        print("Ha shape: ", Ha.shape)
        Ha = torch.squeeze(Ha, dim=1)
        print("Ha shape (2): ", Ha.shape)
        Ha = self.aa_net['fc'](Ha)
        Aa = self.aa_net['relu'](Ha)

        fA = torch.mul(audio, torch.unsqueeze(Aa, dim=1).repeat(1, self.audio_embed_size, 1))
        fA = torch.mean(fA, dim=2)  # [bs x 256]

        # Fusion
        fSCTA = torch.cat([fSCT, fA], dim=1)
        output = self.av_fc(fSCTA)

        if not compute_gradcam:
            return output, alpha, beta, gamma

        if self._gc_activ is None:
            raise RuntimeError("Grad-CAM activation is None. Hook may not be attached correctly.")

        score = output.gather(1, target_class.view(-1, 1)).sum()

        retain = self.training
        #retain_graph: forward를 하면서 연산 그래프를 저장해둘 것인가.
        # 연산 기록을 저장해두면, 그 기록을 따라서 gradient를 계산할 수 있다.
        #create_graph: gradient 계산 과정도 그래프로 기록할 것인가. -> gradient에 대해서도 미분 가능해짐(2차 미분)
        grads = torch.autograd.grad(score, self._gc_activ, retain_graph=retain, create_graph=True)[0] #backward 정보 저장, 2차 미분 없이 일단 세팅

        A = self._gc_activ
        if A.dim() == 3:  # [N, C, m]
            w = grads.mean(dim=2, keepdim=True)          # [N, C, 1]
            cam_vec = torch.relu((w * A).sum(dim=1))     # [N, m]
            hw = int(self.hp['hw'])  # 보통 4
            cam2 = cam_vec.view(-1, hw, hw)              # [N, 4, 4]
        elif A.dim() == 5:
            # [N, C, T, H, W]
            w = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam3 = torch.relu((w * A).sum(dim=1))   # [N, T, H, W]
            cam2 = cam3.mean(dim=1)                 # [N, H, W]
        else:
            w = grads.mean(dim=(2, 3), keepdim=True)
            cam2 = torch.relu((w * A).sum(dim=1))   # [N, H, W]

        # cam2 = cam2 / (cam2.amax(dim=(1, 2), keepdim=True) + 1e-6) #샘플별로 정규화하는 코드

        cam2_up = torch.nn.functional.interpolate(
            cam2.unsqueeze(1),
            size=(sample_size, sample_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, H, W]

        cam_map = cam2_up.view(seq_len, batch, sample_size, sample_size).permute(1, 0, 2, 3).contiguous()  # [B, Seq, H, W]

        return output, alpha, beta, gamma, cam_map
