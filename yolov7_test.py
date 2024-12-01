import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch.nn as nn

from torchvision import transforms
from utils.general import non_max_suppression_kpt
from utils.datasets import letterbox
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def create_nobatSeqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=True)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    conv.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])
    return conv

def create_groupSeqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num):
    conv = nn.Conv2d(oup, oup, kernel_size, stride, pad, groups=oup, bias=False)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    bn = nn.BatchNorm2d(oup, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    bn.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])   
    bn.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 2]])
    bn.running_mean = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 3]])
    bn.running_var = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 4]])
    # bn.num_batches_tracked = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 5]])

    return nn.Sequential(
        conv,
        bn,
        nn.SiLU(inplace=True)
    )

def create_seqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    bn = nn.BatchNorm2d(oup, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    bn.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])   
    bn.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 2]])
    bn.running_mean = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 3]])
    bn.running_var = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 4]])
    # bn.num_batches_tracked = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 5]])

    return nn.Sequential(
        conv,
        bn,
        nn.SiLU(inplace=True)
    )

def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class CSPLayer(nn.Module): # 42
    def __init__(self, input_filter, wgt, wgt_list, wgt_num):
        super().__init__()
        self.conv1 = create_seqs(input_filter * 2, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num)
        self.conv2 = create_seqs(input_filter * 2, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 6)
        
        self.bottle1 = create_seqs(input_filter, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 12)
        self.bottle2 = create_seqs(input_filter, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 18)
        self.bottle3 = create_seqs(input_filter, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 24)
        self.bottle4 = create_seqs(input_filter, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 30)
        
        self.conv3 = create_seqs(input_filter * 4, input_filter * 2, 1, 1, 0, wgt, wgt_list, wgt_num + 36)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x3 = self.bottle1(x2)
        x3 = self.bottle2(x3)
        x4 = self.bottle3(x3)
        x4 = self.bottle4(x4)
        x_cat = torch.concat((x4, x3, x2, x1), 1)
        x_cat = self.conv3(x_cat)
        return x_cat

class NeoCSPLayer(nn.Module): # 42
    def __init__(self, input_filter, wgt, wgt_list, wgt_num):
        super().__init__()
        self.conv1 = create_seqs(input_filter * 2, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num)
        self.conv2 = create_seqs(input_filter * 2, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 6)
        
        self.bottle1 = create_seqs(input_filter, int(input_filter / 2), 3, 1, 1, wgt, wgt_list, wgt_num + 12)
        self.bottle2 = create_seqs(int(input_filter / 2), int(input_filter / 2), 3, 1, 1, wgt, wgt_list, wgt_num + 18)
        self.bottle3 = create_seqs(int(input_filter / 2), int(input_filter / 2), 3, 1, 1, wgt, wgt_list, wgt_num + 24)
        self.bottle4 = create_seqs(int(input_filter / 2), int(input_filter / 2), 3, 1, 1, wgt, wgt_list, wgt_num + 30)   
        self.conv3 = create_seqs(input_filter * 4, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 36)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.bottle1(x2)
        x4 = self.bottle2(x3)
        x5 = self.bottle3(x4)
        x6 = self.bottle4(x5)
        
        x_cat = torch.concat((x6, x5, x4, x3, x2, x1), 1)
        x_cat = self.conv3(x_cat)
        return x_cat

class LayerCircle(nn.Module):
    def __init__(self, input_filter, wgt, wgt_list, wgt_num):
        super().__init__()
        self.conv1 = create_groupSeqs(1, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num)
        self.conv2 = create_seqs(input_filter, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 6)        
        self.conv3 = create_groupSeqs(1, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 12)
        self.conv4 = create_seqs(input_filter, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 18) 
        self.conv5 = create_groupSeqs(1, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 24)
        self.conv6 = create_seqs(input_filter, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 30)
        self.conv7 = create_groupSeqs(1, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 36)
        self.conv8 = create_seqs(input_filter, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 42)
        self.conv9 = create_groupSeqs(1, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 48)
        self.conv10 = create_seqs(input_filter, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 54)
        self.conv11 = create_groupSeqs(1, input_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 60)   
        self.conv12 = create_nobatSeqs(input_filter, 153, 1, 1, 0, wgt, wgt_list, wgt_num + 66) # 68
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        return x

class Implicit(nn.Module):
    def __init__(self, addmul, wgt, wgt_list, wgt_num):
        super().__init__()
        self.a = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
        self.ismul = addmul
    
    def forward(self, x):
        if self.ismul == 0:
            x = x + self.a
        else:
            x = x * self.a

        return x


class Yolov7(nn.Module):
    def __init__(self, wgt, wgt_list):
        super().__init__()  
        self.conv_41 = create_seqs(12, 64, 3, 1, 1, wgt, wgt_list, 0)
        
        self.conv_45 = create_seqs(64, 128, 3, 2, 1, wgt, wgt_list, 6)
        self.csp1 = CSPLayer(64, wgt, wgt_list, 12)

        self.conv_78 = create_seqs(128, 256, 3, 2, 1, wgt, wgt_list, 54)
        self.csp2 = CSPLayer(128, wgt, wgt_list, 60) # shortcut -> 195

        self.conv_111 = create_seqs(256, 512, 3, 2, 1, wgt, wgt_list, 102)
        self.csp3 = CSPLayer(256, wgt, wgt_list, 108) # shortcut -> 287

        self.conv_144 = create_seqs(512, 768, 3, 2, 1, wgt, wgt_list, 150)
        self.csp4 = CSPLayer(384, wgt, wgt_list, 156) # shorcut -> 379

        self.conv_177 = create_seqs(768, 1024, 3, 2, 1, wgt, wgt_list, 198)
        self.csp5 = CSPLayer(512, wgt, wgt_list, 204)

        self.conv_210 = create_seqs(1024, 512, 1, 1, 0, wgt, wgt_list, 246)
        self.conv_234 = create_seqs(1024, 512, 1, 1, 0, wgt, wgt_list, 252)
        self.conv_214 = create_seqs(512, 512, 3, 1, 1, wgt, wgt_list, 258)
        self.conv_218 = create_seqs(512, 512, 1, 1, 0, wgt, wgt_list, 264)

        self.maxpool_222 = nn.MaxPool2d(5, stride=1, padding=2)
        self.maxpool_223 = nn.MaxPool2d(9, stride=1, padding=4)
        self.maxpool_224 = nn.MaxPool2d(13, stride=1, padding=6)

        self.conv_226 = create_seqs(2048, 512, 1, 1, 0, wgt, wgt_list, 270)
        self.conv_230 = create_seqs(512, 512, 3, 1, 1, wgt, wgt_list, 276)
        self.conv_239 = create_seqs(1024, 512, 1, 1, 0, wgt, wgt_list, 282)

        self.conv_243 = create_seqs(512, 384, 1, 1, 0, wgt, wgt_list, 288)
        self.up_248 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_249 = create_seqs(768, 384, 1, 1, 0, wgt, wgt_list, 294)
        self.ncsp1 = NeoCSPLayer(384, wgt, wgt_list, 300)
        
        self.conv_283 = create_seqs(384, 256, 1, 1, 0, wgt, wgt_list, 342)
        self.up_288 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_289 = create_seqs(512, 256, 1, 1, 0, wgt, wgt_list, 348)
        self.ncsp2 = NeoCSPLayer(256, wgt, wgt_list, 354)

        self.conv_323 = create_seqs(256, 128, 1, 1, 0, wgt, wgt_list, 396)
        self.up_328 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_329 = create_seqs(256, 128, 1, 1, 0, wgt, wgt_list, 402)
        self.ncsp3 = NeoCSPLayer(128, wgt, wgt_list, 408)


        self.conv_363 = create_seqs(128, 256, 3, 2, 1, wgt, wgt_list, 450)
        self.ncsp4 = NeoCSPLayer(256, wgt, wgt_list, 456)

        self.conv_397 = create_seqs(256, 384, 3, 2, 1, wgt, wgt_list, 498)
        self.ncsp5 = NeoCSPLayer(384, wgt, wgt_list, 504)

        self.conv_431 = create_seqs(384, 512, 3, 2, 1, wgt, wgt_list, 546)
        self.ncsp6 = NeoCSPLayer(512, wgt, wgt_list, 552)

        self.conv_465 = create_seqs(128, 256, 3, 1, 1, wgt, wgt_list, 594)
        self.conv_469 = create_seqs(256, 512, 3, 1, 1, wgt, wgt_list, 600)
        self.conv_473 = create_seqs(384, 768, 3, 1, 1, wgt, wgt_list, 606)
        self.conv_477 = create_seqs(512, 1024, 3, 1, 1, wgt, wgt_list, 612)
        # /model.118.anchors : torch.Size([4, 3, 2]) 618
        # /model.118.anchor_grid : torch.Size([4, 1, 3, 1, 1, 2]) 619

        self.c0 = create_nobatSeqs(256, 18, 1, 1, 0, wgt, wgt_list, 620)
        self.c1 = create_nobatSeqs(256, 18, 1, 1, 0, wgt, wgt_list, 622)
        self.c2 = create_nobatSeqs(256, 18, 1, 1, 0, wgt, wgt_list, 624)
        self.c3 = create_nobatSeqs(256, 18, 1, 1, 0, wgt, wgt_list, 626)
        self.a0 = Implicit(0, wgt, wgt_list, 628)
        self.a1 = Implicit(0, wgt, wgt_list, 629)
        self.a2 = Implicit(0, wgt, wgt_list, 630)
        self.a3 = Implicit(0, wgt, wgt_list, 631)
        
        self.m0 = Implicit(1, wgt, wgt_list, 632)
        self.m1 = Implicit(1, wgt, wgt_list, 633)
        self.m2 = Implicit(1, wgt, wgt_list, 634)
        self.m3 = Implicit(1, wgt, wgt_list, 635)

        self.circle1 = LayerCircle(256, wgt, wgt_list, 636)
        self.circle2 = LayerCircle(512, wgt, wgt_list, 704)
        self.circle3 = LayerCircle(768, wgt, wgt_list, 772)
        self.circle4 = LayerCircle(1024, wgt, wgt_list, 840) # 908


    def forward(self, x):
        x0 = self.conv_41(x)
        x0 = self.conv_45(x0)
        x0 = self.csp1(x0)

        x0 = self.conv_78(x0)
        x0 = self.csp2(x0) # div

        x1 = self.conv_111(x0)
        x1 = self.csp3(x1) # div
        
        x2 = self.conv_144(x1)
        x2 = self.csp4(x2) # div

        x3 = self.conv_177(x2)
        x3 = self.csp5(x3)
        #
        x3_c = self.conv_234(x3)
        x3 = self.conv_218(self.conv_214(self.conv_210(x3)))
        
        x3_1 = self.maxpool_222(x3)
        x3_2 = self.maxpool_223(x3)
        x3_3 = self.maxpool_224(x3)
        
        x3 = torch.concat((x3, x3_1, x3_2, x3_3), 1)
        x3 = self.conv_230(self.conv_226(x3))
        x3 = torch.concat((x3, x3_c), 1) # 238
        x3 = self.conv_239(x3) # div
        #
        x4 = self.conv_243(x3)
        x4 = self.up_248(x4)
        x2 = self.conv_249(x2)
        x4 = torch.concat((x2, x4), 1)
        x4 = self.ncsp1(x4) # div
        
        x5 = self.conv_283(x4)
        x5 = self.up_288(x5)
        x1 = self.conv_289(x1)
        x5 = torch.concat((x1, x5), 1)
        x5 = self.ncsp2(x5)

        x6 = self.conv_323(x5)
        x6 = self.up_328(x6)
        x0 = self.conv_329(x0)
        x6 = torch.concat((x0, x6), 1)
        x6 = self.ncsp3(x6) # shortcut

        x7 = self.conv_363(x6)
        x7 = torch.concat((x7, x5), 1)
        x7 = self.ncsp4(x7) # shortcut

        x8 = self.conv_397(x7)
        x8 = torch.concat((x8, x4), 1)
        x8 = self.ncsp5(x8) # shortcut

        x9 = self.conv_431(x8)
        x9 = torch.concat((x9, x3), 1)
        x9 = self.ncsp6(x9) # shortcut

        x6 = self.conv_465(x6)
        x7 = self.conv_469(x7)
        x8 = self.conv_473(x8)
        x9 = self.conv_477(x9)
        
        x6_a = self.circle1(x6)
        x6_b = self.a0(x6)
        x6_b = self.c0(x6_b)
        x6_b = self.m0(x6_b)
        x6 = torch.concat((x6_b, x6_a), 1)

        x7_a = self.circle2(x7)
        x7_b = self.a1(x7)
        x7_b = self.c1(x7_b)
        x7_b = self.m1(x7_b)
        x7 = torch.concat((x7_b, x7_a), 1)

        x8_a = self.circle3(x8)
        x8_b = self.a2(x8)
        x8_b = self.c2(x8_b)
        x8_b = self.m2(x8_b)
        x8 = torch.concat((x8_b, x8_a), 1)

        x9_a = self.circle4(x9)
        x9_b = self.a3(x9)
        x9_b = self.c3(x9_b)
        x9_b = self.m3(x9_b)
        x9 = torch.concat((x9_b, x9_a), 1)

        out_list = [x6, x7, x8, x9]
        grid = torch.Tensor([0.0])
        num = 0

        z = []
        stride_list = [8, 16, 32, 64]
        anchor_list = [torch.Tensor([[[[[19., 27.]]], [[[44., 40.]]], [[[38., 94.]]]]]).to("cuda"), torch.Tensor([[[[[ 96.,  68.]]], [[[ 86., 152.]]], [[[180., 137.]]]]]).to("cuda"),
            torch.Tensor([[[[[140., 301.]]], [[[303., 264.]]], [[[238., 542.]]]]]).to("cuda"), torch.Tensor([[[[[436., 615.]]], [[[739., 380.]]], [[[925., 792.]]]]]).to("cuda")]

        for out in out_list:
            bs, _, ny, nx = out.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            out = out.view(bs, 3, 57, ny, nx)
            out = out.permute(0, 1, 3, 4, 2).contiguous()
            # out = out.permute(0, 2, 3, 1).contiguous()
            x_det = out[..., :6]
            x_kpt = out[..., 6:]
            
            if grid.shape[2:4] != out.shape[2:4]:
                grid = _make_grid(nx, ny).to(out.device)

            kpt_grid_x = grid[..., 0:1]
            kpt_grid_y = grid[..., 1:2]

            y = x_det.sigmoid()
            xy = (y[..., 0:2] * 2. - 0.5 + grid) * stride_list[num]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * anchor_list[num].view(1, 3, 1, 1, 2) # wh

            x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * stride_list[num]  # xy
            x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * stride_list[num]  # xy
            x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

            y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)
            # print(torch.sum(y))
            z.append(y.view(bs, -1, 57))
            num += 1
        
        return (torch.cat(z, 1), x)

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 0
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 0

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    orig_image = cv2.imread("../TestImage/people_test.jpg")
    # pre, _ = preproc(orig_image, (640, 640))
    
    orig_image = cv2.resize(orig_image, (640, 640))
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # image = letterbox(image, (orig_image.shape[1]), stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    
    image.to(device)
    weight = torch.load("weights/yolov7.pth") # 가중치 경로
    wgt_list = list(weight.keys())
    test_model = Yolov7(weight, wgt_list)
    test_model.float().to(device)
    test_model.eval()
    
    patch_top_left = image[..., ::2, ::2]
    patch_bot_left = image[..., 1::2, ::2]
    patch_top_right = image[..., ::2, 1::2]
    patch_bot_right = image[..., 1::2, 1::2]

    img = torch.cat(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        dim=1,
    )

    img = img.to(device)
    
    with torch.no_grad():
        output, _ = test_model(img)
        # torch.save(test_model.state_dict(), "yolov7pure.pth")

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
    print(output[0].shape)
    output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

        xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
        xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
        cv2.rectangle(
            nimg,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        
    cv2.imshow('result', nimg)
    cv2.waitKey(0)
        