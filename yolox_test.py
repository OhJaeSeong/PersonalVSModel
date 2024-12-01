import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import numpy as np
# from onnx import numpy_helper
import cv2

import torch.nn.functional as F

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]

def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)

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

def create_nobatSeqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=True)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    conv.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])
    return conv

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


class bottleneck_list(nn.Module):
    def __init__(self, filter, shortcut, wgt, wgt_list, wgt_num):
        super().__init__()

        self.shortcut = shortcut
        self.conv1 = create_seqs(filter, filter, 1, 1, 0, wgt, wgt_list, wgt_num)
        self.conv2 = create_seqs(filter, filter, 3, 1, 1, wgt, wgt_list, wgt_num + 6)
        self.conv3 = create_seqs(filter, filter, 1, 1, 0, wgt, wgt_list, wgt_num + 12)
        self.conv4 = create_seqs(filter, filter, 3, 1, 1, wgt, wgt_list, wgt_num + 18)

    def forward(self, x):
        if self.shortcut:
            x1 = self.conv2(self.conv1(x))
            x2 = x + x1
            x3 = self.conv4(self.conv3(x2))
            x4 = x2 + x3
            return x4
        else:
            x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
            return x
        

class CSPLayer(nn.Module):
    def __init__(self, input_filter, output_filter, count, isAdd, wgt, wgt_list, wgt_num):
        super().__init__()

        self.number = count
        self.conv1 = create_seqs(input_filter, int(output_filter/2), 1, 1, 0, wgt, wgt_list, wgt_num)
        self.conv2 = create_seqs(input_filter, int(output_filter/2), 1, 1, 0, wgt, wgt_list, wgt_num + 6)
        self.conv3 = create_seqs(output_filter, int(output_filter), 1, 1, 0, wgt, wgt_list, wgt_num + 12)

        self.bottleneck1 = bottleneck_list(int(output_filter/2), isAdd, wgt, wgt_list, wgt_num + 18) # 42
        
        if count == 3:
            self.bottleneck2 = bottleneck_list(int(output_filter/2), isAdd, wgt, wgt_list, wgt_num + 42)
            self.bottleneck3 = bottleneck_list(int(output_filter/2), isAdd, wgt, wgt_list, wgt_num + 66) # 90

    def forward(self, x):
        x1 = self.bottleneck1(self.conv1(x))
        if self.number > 1:
            x1 = self.bottleneck3(self.bottleneck2(x1))
        
        x2 = self.conv2(x)
        x_cat = torch.concat((x1, x2), 1)
        x_cat = self.conv3(x_cat)
        return x_cat

class YoloX(nn.Module):
    def __init__(self, wgt, wgt_list):
        super().__init__()
        
        self.conv_41 = create_seqs(12, 48, 3, 1, 1, wgt, wgt_list, 0)
        self.conv_44 = create_seqs(48, 96, 3, 2, 1, wgt, wgt_list, 6) # dark2
        self.csp1 = CSPLayer(96, 96, 1, True, wgt, wgt_list, 12)

        self.conv_71 = create_seqs(96, 192, 3, 2, 1, wgt, wgt_list, 54)
        self.csp2 = CSPLayer(192, 192, 3, True, wgt, wgt_list, 60)

        self.conv_126 = create_seqs(192, 384, 3, 2, 1, wgt, wgt_list, 150)
        self.csp3 = CSPLayer(384, 384, 3, True, wgt, wgt_list, 156)

        self.conv_181 = create_seqs(384, 768, 3, 2, 1, wgt, wgt_list, 246)
        self.conv_184 = create_seqs(768, 384, 1, 1, 0, wgt, wgt_list, 252)

        self.maxPool_187 = nn.MaxPool2d(5, stride=1, padding=2)
        self.maxPool_188 = nn.MaxPool2d(9, stride=1, padding=4)
        self.maxPool_189 = nn.MaxPool2d(13, stride=1, padding=6)

        self.conv_191 = create_seqs(1536, 768, 1, 1, 0, wgt, wgt_list, 258)
        self.csp4 = CSPLayer(768, 768, 1, False, wgt, wgt_list, 264)

        self.conv_216 = create_seqs(768, 384, 1, 1, 0, wgt, wgt_list, 306)
        self.up_x0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.csp5 = CSPLayer(768, 384, 1, False, wgt, wgt_list, 312)

        self.conv_244 = create_seqs(384, 192, 1, 1, 0, wgt, wgt_list, 354)
        self.up_x1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.csp6 = CSPLayer(384, 192, 1, False, wgt, wgt_list, 360)

        self.conv_272 = create_seqs(192, 192, 3, 2, 1, wgt, wgt_list, 402)
        self.csp7 = CSPLayer(384, 384, 1, False, wgt, wgt_list, 408)

        self.conv_298 = create_seqs(384, 384, 3, 2, 1, wgt, wgt_list, 450)
        self.csp8 = CSPLayer(768, 768, 1, False, wgt, wgt_list, 456)

        

        self.cls_convs1_1 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 498)
        self.cls_convs1_2 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 504)
        
        self.cls_convs2_1 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 510)
        self.cls_convs2_2 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 516)

        self.cls_convs3_1 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 522)
        self.cls_convs3_2 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 528)


        self.reg_convs1_1 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 534)
        self.reg_convs1_2 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 540)
        
        self.reg_convs2_1 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 546)
        self.reg_convs2_2 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 552)

        self.reg_convs3_1 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 558)
        self.reg_convs3_2 = create_seqs(192, 192, 3, 1, 1, wgt, wgt_list, 564)


        self.cls_preds1 = create_nobatSeqs(192, 80, 1, 1, 0, wgt, wgt_list, 570)
        self.cls_preds2 = create_nobatSeqs(192, 80, 1, 1, 0, wgt, wgt_list, 572)
        self.cls_preds3 = create_nobatSeqs(192, 80, 1, 1, 0, wgt, wgt_list, 574)

        self.reg_preds1 = create_nobatSeqs(192, 4, 1, 1, 0, wgt, wgt_list, 576)
        self.reg_preds2 = create_nobatSeqs(192, 4, 1, 1, 0, wgt, wgt_list, 578)
        self.reg_preds3 = create_nobatSeqs(192, 4, 1, 1, 0, wgt, wgt_list, 580)

        self.obj_preds1 = create_nobatSeqs(192, 1, 1, 1, 0, wgt, wgt_list, 582)
        self.obj_preds2 = create_nobatSeqs(192, 1, 1, 1, 0, wgt, wgt_list, 584)
        self.obj_preds3 = create_nobatSeqs(192, 1, 1, 1, 0, wgt, wgt_list, 586)

        # stem
        self.stem1 = create_seqs(192, 192, 1, 1, 0, wgt, wgt_list, 588)
        self.stem2 = create_seqs(384, 192, 1, 1, 0, wgt, wgt_list, 594)
        self.stem3 = create_seqs(768, 192, 1, 1, 0, wgt, wgt_list, 600)

        self.sig = nn.Sigmoid()


    def forward(self, x):
        # backbone(YoloFPN)
        # - Focus
        x = self.conv_41(x)
        # - CSPDarknet
        x = self.csp1(self.conv_44(x))
        x = self.csp2(self.conv_71(x))
        p1 = x
        
        x = self.csp3(self.conv_126(x))
        p2 = x
        
        x = self.conv_184(self.conv_181(x))    
        m1 = self.maxPool_187(x)
        m2 = self.maxPool_188(x)
        m3 = self.maxPool_189(x)
        
        x = torch.concat((x, m1, m2, m3), 1)
        x = self.csp4(self.conv_191(x))
        # - CSPDarknet End
        
        # lateral_conv0
        p3 = self.conv_216(x) # ->
        up1 = self.up_x0(p3)
        p2 = torch.concat((up1, p2), 1)

        # C3_p4
        p2 = self.csp5(p2)

        # reduce_conv1
        p2 = self.conv_244(p2) # ->

        up2 = self.up_x1(p2)
        p1 = torch.concat((up2, p1), 1)
        # C3_p3
        p1 = self.csp6(p1)
        
        # bu_conv2
        p1_1 = self.conv_272(p1)  
        p2 = torch.concat((p1_1, p2), 1)
        # C3_n3
        p2 = self.csp7(p2)     
        
        # bu_conv1
        p2_1 = self.conv_298(p2)
        p3 = torch.concat((p2_1, p3), 1)
        # C3_n4
        p3 = self.csp8(p3)

        # head
        p1 = self.stem1(p1)
        reg_out1 = self.reg_convs1_2(self.reg_convs1_1(p1))
        obj_out1 = self.obj_preds1(reg_out1)
        reg_out1 = self.reg_preds1(reg_out1)
        
        cls_out1 = self.cls_convs1_2(self.cls_convs1_1(p1))
        cls_out1 = self.cls_preds1(cls_out1)

        p2 = self.stem2(p2)
        reg_out2 = self.reg_convs2_2(self.reg_convs2_1(p2))
        obj_out2 = self.obj_preds2(reg_out2)
        reg_out2 = self.reg_preds2(reg_out2)

        cls_out2 = self.cls_convs2_2(self.cls_convs2_1(p2))
        cls_out2 = self.cls_preds2(cls_out2)


        p3 = self.stem3(p3)
        reg_out3 = self.reg_convs3_2(self.reg_convs3_1(p3))
        obj_out3 = self.obj_preds3(reg_out3)
        reg_out3 = self.reg_preds3(reg_out3)

        cls_out3 = self.cls_convs3_2(self.cls_convs3_1(p3))
        cls_out3 = self.cls_preds3(cls_out3)

        out1 = torch.concat((reg_out1, self.sig(obj_out1), self.sig(cls_out1)), 1)
        out2 = torch.concat((reg_out2, self.sig(obj_out2), self.sig(cls_out2)), 1)
        out3 = torch.concat((reg_out3, self.sig(obj_out3), self.sig(cls_out3)), 1)
        
        outputs = [out1, out2, out3]
        hw = [x.shape[-2:] for x in outputs]

        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        outputs = self.decode_outputs(outputs, hw, out1.type())
        return outputs

    def decode_outputs(self, outputs, hw, dtype):
        grids = []
        st = [8, 16, 32]
        strides = []
        for (hsize, wsize), stride in zip(hw, st):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def transform(img, input_size):
    img, _ = preproc(img, input_size)
    return img


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction[:, :, :4]
    box_corner = torch.pow(box_corner, 1)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True) # class_conf : 값, pred : 값의 위치
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        if not detections.size(0):
            continue

        if class_agnostic:
            detections = detections.cpu().numpy()
            dets = np.zeros((detections.shape[0], 5))
            dets[: , 0:4] = detections[ : , 0:4]
            dets[:, 4] = detections[:, 4] * detections[:, 5]
            nms_out_index = py_cpu_nms(dets, 0.4)
            '''
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )'''
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

if __name__ == "__main__":
    weight = torch.load("weights/yolox_normal_forAll.pth") # 가중치 경로
    wgt_list = list(weight.keys())
    yolox = YoloX(weight, wgt_list)
    yolox.eval()

    image = cv2.imread("../TestImage/people_test.jpg")
    resize_num = 640

    ratio = min(resize_num/image.shape[1], resize_num/image.shape[0])
    img = transform(image, (resize_num, resize_num))
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()

    patch_top_left = img[..., ::2, ::2]
    patch_top_right = img[..., ::2, 1::2]
    patch_bot_left = img[..., 1::2, ::2]
    patch_bot_right = img[..., 1::2, 1::2]
    img = torch.cat(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        dim=1,
    )

    yolox.to("cuda")
    img = img.to("cuda")
    with torch.no_grad():
        outputs = yolox(img)
        # torch.save(yolox, "yoloxpure.pth")

    outputs = postprocess(
                outputs, 80, 0.4,
                0.45, class_agnostic=False
    )
    
    if True:
        outputs = outputs[0]
        for k in range(0, len(outputs)):
            i = outputs[k]
            x1 = int(i[0] / ratio)
            y1 = int(i[1] / ratio)
            x2 = int(i[2] / ratio)
            y2 = int(i[3] / ratio)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("img", image)
    cv2.waitKey(0)

    # torch.save(yolox.state_dict(), "yolox_wgt.weights")
