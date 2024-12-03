from turtle import forward
from venv import create
import cv2

from itertools import product as product
import numpy as np
from priorbox import PriorBox
import torch.nn.functional as F

import torch
import torch.nn as nn


def create_seqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num, act):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    bn = nn.BatchNorm2d(oup, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])   
    bn.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 2]])
    bn.running_mean = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 3]])
    bn.running_var = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 4]])
    # bn.num_batches_tracked = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 5]])

    if act == 0:
        return nn.Sequential(
            conv,
            bn,
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            conv,
            bn,
            nn.LeakyReLU(negative_slope=0, inplace=True)
        )

def create_NoactSeqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    bn = nn.BatchNorm2d(oup, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])   
    bn.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 2]])
    bn.running_mean = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 3]])
    bn.running_var = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 4]])
    # bn.num_batches_tracked = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 5]])

    return nn.Sequential(
            conv,
            bn
    )

def biasSeqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=True)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    conv.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])
    return conv
    

class BottleneckLoop(nn.Module):
    def __init__(self, inp, oup, wgt, wgt_list, wgt_num):
        super().__init__()
        self.conv1 = create_seqs(inp, oup, 1, 1, 0, wgt, wgt_list, wgt_num, 0)
        self.conv2 = create_seqs(oup, oup, 3, 1, 1, wgt, wgt_list, wgt_num + 6, 0)
        self.conv3 = create_NoactSeqs(oup, inp, 1, 1, 0, wgt, wgt_list, wgt_num + 12)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x1 = self.conv3(self.conv2(self.conv1(x)))
        return self.relu(x1 + x)

class BottleneckDLoop(nn.Module):
    def __init__(self, inp, oup, wgt, wgt_list, wgt_num):
        super().__init__()
        self.conv1 = create_seqs(int(inp / 2), oup, 1, 1, 0, wgt, wgt_list, wgt_num, 0)
        self.conv2 = create_seqs(oup, oup, 3, 2, 1, wgt, wgt_list, wgt_num + 6, 0)
        self.conv3 = create_NoactSeqs(oup, inp, 1, 1, 0, wgt, wgt_list, wgt_num + 12)
        self.conv_down = create_NoactSeqs(oup * 2, inp, 1, 2, 0, wgt, wgt_list, wgt_num + 18)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x1 = self.conv3(self.conv2(self.conv1(x)))
        x2 = self.conv_down(x)
        return self.relu(x1 + x2)

class FPN(nn.Module):
    def __init__(self, inp, oup, wgt, wgt_list, wgt_num):
        super().__init__()
        self.conv1 = create_NoactSeqs(inp, oup * 2, 3, 1, 1, wgt, wgt_list, wgt_num)
        self.conv2 = create_seqs(inp, oup, 3, 1, 1, wgt, wgt_list, wgt_num + 6, 1)
        self.conv3 = create_NoactSeqs(oup, oup, 3, 1, 1, wgt, wgt_list, wgt_num + 12)
        self.conv4 = create_seqs(oup, oup, 3, 1, 1, wgt, wgt_list, wgt_num + 18, 1)
        self.conv5 = create_NoactSeqs(oup, oup, 3, 1, 1, wgt, wgt_list, wgt_num + 24)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv5(self.conv4(x2))
        x2 = self.conv3(x2)

        x_cat = torch.cat((x1, x2, x3), 1)
        return self.relu(x_cat)


class HBRetinaFace(nn.Module):
    def __init__(self, wgt, wgt_list):
        super().__init__()
        self.conv_0 = create_seqs(3, 64, 7, 2, 3, wgt, wgt_list, 0, 0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # circle1
        self.conv_3 = create_seqs(64, 64, 1, 1, 0, wgt, wgt_list, 6, 0)
        self.conv_5 = create_seqs(64, 64, 3, 1, 1, wgt, wgt_list, 12, 0)
        self.conv_7 = create_NoactSeqs(64, 256, 1, 1, 0, wgt, wgt_list, 18)
        self.conv_8 = create_NoactSeqs(64, 256, 1, 1, 0, wgt, wgt_list, 24)
        
        self.circle2 = BottleneckLoop(256, 64, wgt, wgt_list, 30)
        self.circle3 = BottleneckLoop(256, 64, wgt, wgt_list, 48)

        self.circle4 = BottleneckDLoop(512, 128, wgt, wgt_list, 66)
        
        self.circle5 = BottleneckLoop(512, 128, wgt, wgt_list, 90)
        # self.circle6 = BottleneckLoop(512, 128, wgt, wgt_list, 108)
        self.circle7 = BottleneckLoop(512, 128, wgt, wgt_list, 126) # ->

        self.circle8 = BottleneckDLoop(1024, 256, wgt, wgt_list, 144)
        # self.circle9 = BottleneckLoop(1024, 256, wgt, wgt_list, 168)
        # self.circle10 = BottleneckLoop(1024, 256, wgt, wgt_list, 186)
        # self.circle11 = BottleneckLoop(1024, 256, wgt, wgt_list, 204)
        self.circle12 = BottleneckLoop(1024, 256, wgt, wgt_list, 222)
        self.circle13 = BottleneckLoop(1024, 256, wgt, wgt_list, 240) # ->

        self.circle14 = BottleneckDLoop(2048, 512, wgt, wgt_list, 258)
        # self.circle15 = BottleneckLoop(2048, 512, wgt, wgt_list, 282)
        self.circle16 = BottleneckLoop(2048, 512, wgt, wgt_list, 300) #318

        self.conv_119 = create_seqs(512, 256, 1, 1, 0, wgt, wgt_list, 318, 1)
        self.conv_121 = create_seqs(1024, 256, 1, 1, 0, wgt, wgt_list, 324, 1)
        self.conv_123 = create_seqs(2048, 256, 1, 1, 0, wgt, wgt_list, 330, 1) # fpn

        self.conv_163 = create_seqs(256, 256, 3, 1, 1, wgt, wgt_list, 336, 1) # merge
        self.conv_143 = create_seqs(256, 256, 3, 1, 1, wgt, wgt_list, 342, 1) # 348 + 108 = 456

        #SSH
        self.fpn1 = FPN(256, 64, wgt, wgt_list, 348)
        self.fpn2 = FPN(256, 64, wgt, wgt_list, 378)
        self.fpn3 = FPN(256, 64, wgt, wgt_list, 408)

        self.conv_217 = biasSeqs(256, 4, 1, 1, 0, wgt, wgt_list, 438)
        self.conv_225 = biasSeqs(256, 4, 1, 1, 0, wgt, wgt_list, 440)
        self.conv_233 = biasSeqs(256, 4, 1, 1, 0, wgt, wgt_list, 442)

        self.conv_192 = biasSeqs(256, 8, 1, 1, 0, wgt, wgt_list, 444)
        self.conv_200 = biasSeqs(256, 8, 1, 1, 0, wgt, wgt_list, 446)
        self.conv_208 = biasSeqs(256, 8, 1, 1, 0, wgt, wgt_list, 448)

        # self.conv_242 = biasSeqs(256, 20, 1, 1, 0, wgt, wgt_list, 450)
        # self.conv_250 = biasSeqs(256, 20, 1, 1, 0, wgt, wgt_list, 452)
        # self.conv_258 = biasSeqs(256, 20, 1, 1, 0, wgt, wgt_list, 454)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    
    def forward(self, x):
        x = self.conv_0(x)
        x = self.maxpool2(x)
        xa = self.conv_8(x)
        x = self.conv_7(self.conv_5(self.conv_3(x)))
        x = self.relu(x + xa)
        x = self.circle2(x) # circle 2 ~ 3
        x = self.circle3(x)
        x1 = self.circle4(x)
        x1 = self.circle5(x1)
        # x1 = self.circle6(x1)
        x1 = self.circle7(x1)
        x2 = self.circle8(x1)
        # x2 = self.circle11(self.circle10(self.circle9(x2)))
        x2 = self.circle12(x2)
        x2 = self.circle13(x2)

        x3 = self.circle14(x2)
        # x3 = self.circle15(x3)
        x3 = self.circle16(x3)

        x1 = self.conv_119(x1)
        x2 = self.conv_121(x2)
        x3 = self.conv_123(x3)

        x3_out = self.fpn3(x3)
        x3_u = self.upsample(x3)
        x2 = x2 + x3_u
        x2 = self.conv_143(x2)
        x2_out = self.fpn2(x2)
        
        x2_u = self.upsample(x2)
        x1 = x1 + x2_u
        x1 = self.conv_163(x1)
        x1_out = self.fpn1(x1)
        
        x1_1 = self.conv_217(x1_out)
        x1_1 = x1_1.permute(0,2,3,1).contiguous()
        x1_1 = x1_1.view(x1_1.shape[0], -1, 2)

        x1_2 = self.conv_225(x2_out)
        x1_2 = x1_2.permute(0,2,3,1).contiguous()
        x1_2 = x1_2.view(x1_2.shape[0], -1, 2)

        x1_3 = self.conv_233(x3_out)
        x1_3 = x1_3.permute(0,2,3,1).contiguous()
        x1_3 = x1_3.view(x1_3.shape[0], -1, 2)
        cls = F.softmax(torch.cat([x1_1, x1_2, x1_3], dim=1), dim=-1)


        x2_1 = self.conv_192(x1_out)
        x2_1 = x2_1.permute(0,2,3,1).contiguous()
        x2_1 = x2_1.view(x2_1.shape[0], -1, 4)

        x2_2 = self.conv_200(x2_out)
        x2_2 = x2_2.permute(0,2,3,1).contiguous()
        x2_2 = x2_2.view(x2_2.shape[0], -1, 4)
        
        x2_3 = self.conv_208(x3_out)
        x2_3 = x2_3.permute(0,2,3,1).contiguous()
        x2_3 = x2_3.view(x2_3.shape[0], -1, 4)
        bbox = torch.cat([x2_1, x2_2, x2_3], dim=1)
        
        '''
        x3_1 = self.conv_242(x1_out)
        x3_1 = x3_1.permute(0,2,3,1).contiguous()
        x3_1 = x3_1.view(x3_1.shape[0], -1, 10)

        x3_2 = self.conv_250(x2_out)
        x3_2 = x3_2.permute(0,2,3,1).contiguous()
        x3_2 = x3_2.view(x3_2.shape[0], -1, 10)

        x3_3 = self.conv_258(x3_out)
        x3_3 = x3_3.permute(0,2,3,1).contiguous()
        x3_3 = x3_3.view(x3_3.shape[0], -1, 10)
        land = torch.cat([x3_1, x3_2, x3_3], dim=1)'''
        return bbox, cls


def decode(loc, priors, variances):
    boxes = None
    if isinstance(loc, torch.Tensor) and isinstance(priors, torch.Tensor) :
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)

    elif isinstance(loc, np.ndarray) and isinstance(priors, np.ndarray):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)

    else:
        print(type(loc), type(priors))
        print(TypeError("ERROR: INVALID TYPE OF BOUNDING BOX"))

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    landms = None
    if isinstance(pre, torch.Tensor) and isinstance(priors, torch.Tensor):
        landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)

    elif isinstance(pre, np.ndarray) and isinstance(priors, np.ndarray):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), axis=1)

    else:
        print(TypeError("ERROR: INVALID TYPE OF LANDMARKS"))
    
    return landms


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def post_processing(net, img):
    resize = 1
    with torch.no_grad():
        loc, conf = net(img)  # forward pass,  landms 삭제
    loc = loc.float().to("cpu")
    conf = conf.float().to("cpu")
    print(torch.sum(loc), torch.sum(conf))
    # landms = landms.to("cpu")

    _, _, im_height, im_width= img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    priorbox = PriorBox(image_size=(im_height, im_width))
    priors = priorbox.forward()

    priors.to('cpu')
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    # landms = decode_landm(landms.data.squeeze(0), prior_data, [0.1, 0.2])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    
    scale1 = scale1.to('cpu')
    # landms = landms * scale1 / resize
    # landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > 0.5)[0]
    boxes = boxes[inds]
    # landms = landms[inds]
    scores = scores[inds]
    
    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    # landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.3)

    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    # landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:750, :]
    # landms = landms[:750, :]

    # dets = np.concatenate((dets), axis=1)
    return dets


def draw_on(img, faces):
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.astype(np.int)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)

    return dimg


if __name__ == "__main__":
    img_raw = cv2.imread('C:/getty.jpg', cv2.IMREAD_COLOR)
    img_raw = cv2.resize(img_raw, (640, 640))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # img_raw = cv2.resize(img_raw, (1280, 720))
    img = np.float32(img_raw)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.half().to(device)

    weight = torch.load("C:/Resnet50_Final.pth") # 가중치 경로
    wgt_list = list(weight.keys())

    retina = HBRetinaFace(weight, wgt_list)
    retina = retina.half()
    retina.eval()

    dets = post_processing(retina, img)
    dets = list(dets)
    # torch.save((retina.half()).state_dict(), "justtest.pth")
    with torch.no_grad():
        torch.onnx.export(retina,               # 실행될 모델
            img,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
            "16RF_Remaster.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
            export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
            opset_version=11,          # 모델을 변환할 때 사용할 ONNX 버전
            do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
            input_names = ['input'],   # 모델의 입력값을 가리키는 이름
            output_names = ['output'], # 모델의 출력값을 가리키는 이름
            dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                        'output' : {0 : 'batch_size'}})

    dimg = draw_on(img_raw, dets)

    while(True):
        cv2.imshow("res", dimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # with torch.no_grad():
    #     retina(img)
    # img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)