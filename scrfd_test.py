import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch.nn as nn

def nms(dets):
        thresh = 0.4
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

def anchor_center(anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

def make_anchor(stride, size):
    leng = int(pow((size/stride), 2) * 2)
    anchors = torch.zeros(leng ,2)
    loop = int(size/stride)
    for l in range(0, loop):
        for r in range(0, loop):
            anchors[2*l*loop + 2*r, 0] = r * stride
            anchors[2*l*loop + 2*r, 1] = l * stride
            anchors[2*l*loop + 2*r + 1, 0] = r * stride
            anchors[2*l*loop + 2*r + 1, 1] = l * stride

    return anchors

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)

def create_nobatSeqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=True)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    conv.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])
    return conv

def create_seqs(inp, oup, kernel_size, stride, pad, wgt, wgt_list, wgt_num):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False)
    conv.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num]])
    bn = nn.BatchNorm2d(oup, eps=1e-5, momentum=0.01, affine=True, track_running_stats=True)
    bn.weight = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 1]])   
    bn.bias = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 2]])
    bn.running_mean = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 3]])
    bn.running_var = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 4]])
    # bn.num_batches_tracked = nn.parameter.Parameter(wgt[wgt_list[wgt_num + 5]])

    return nn.Sequential(
        conv,
        bn
    )

class Circle(nn.Module):
    def __init__(self, input_filter, mid_filter, wgt, wgt_list, wgt_num): # 18
        super().__init__()
        self.conv1 = create_seqs(input_filter, mid_filter, 1, 1, 0, wgt, wgt_list, wgt_num)
        self.conv2 = create_seqs(mid_filter, mid_filter, 3, 1, 1, wgt, wgt_list, wgt_num + 6)
        self.conv3 = create_seqs(mid_filter, input_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 12)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        xm = self.relu(self.conv1(x))
        xm = self.relu(self.conv2(xm))
        xm = self.conv3(xm)
        return self.relu(x + xm)


class ExpCircle(nn.Module):
    def __init__(self, input_filter, mid_filter, output_filter, str, wgt, wgt_list, wgt_num): # 24
        super().__init__()
        self.conv1 = create_seqs(input_filter, mid_filter, 1, 1, 0, wgt, wgt_list, wgt_num)
        self.conv2 = create_seqs(mid_filter, mid_filter, 3, str, 1, wgt, wgt_list, wgt_num + 6)
        self.conv3 = create_seqs(mid_filter, output_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 12)
        self.avg = nn.AvgPool2d(str, stride=str, padding=0)
        self.conv4 = create_seqs(input_filter, output_filter, 1, 1, 0, wgt, wgt_list, wgt_num + 18)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        x1 = self.conv3(x1)
        
        x2 = self.avg(x)
        x2 = self.conv4(x2)
        return self.relu(x1 + x2)


class Scrfd(nn.Module):
    def __init__(self, wgt, wgt_list):
        super().__init__()  
        self.conv_0 = create_seqs(3, 28, 3, 2, 1, wgt, wgt_list, 0)
        self.conv_2 = create_seqs(28, 28, 3, 1, 1, wgt, wgt_list, 6)
        self.conv_4 = create_seqs(28, 56, 3, 1, 1, wgt, wgt_list, 12)
        self.maxpool_6 = nn.MaxPool2d(2, stride=2, padding=0)

        self.expc1 = ExpCircle(56, 56, 224, 1, wgt, wgt_list, 18)
        self.circle1 = Circle(56, 224, wgt, wgt_list, 42)
        self.circle2 = Circle(56, 224, wgt, wgt_list, 60)
        self.circle3 = Circle(56, 224, wgt, wgt_list, 78)
        self.circle4 = Circle(56, 224, wgt, wgt_list, 96)
        self.circle5 = Circle(56, 224, wgt, wgt_list, 114)
        self.circle6 = Circle(56, 224, wgt, wgt_list, 132)
        self.circle7 = Circle(56, 224, wgt, wgt_list, 150)
        self.circle8 = Circle(56, 224, wgt, wgt_list, 168)
        self.circle9 = Circle(56, 224, wgt, wgt_list, 186)
        self.circle10 = Circle(56, 224, wgt, wgt_list, 204)
        self.circle11 = Circle(56, 224, wgt, wgt_list, 222)
        self.circle12 = Circle(56, 224, wgt, wgt_list, 240)
        self.circle13 = Circle(56, 224, wgt, wgt_list, 258)
        self.circle14 = Circle(56, 224, wgt, wgt_list, 276)
        self.circle15 = Circle(56, 224, wgt, wgt_list, 294)
        self.circle16 = Circle(56, 224, wgt, wgt_list, 312)

        self.expc2 = ExpCircle(224, 56, 224, 2, wgt, wgt_list, 330)
        self.circle17 = Circle(56, 224, wgt, wgt_list, 354)
        self.circle18 = Circle(56, 224, wgt, wgt_list, 372)
        self.circle19 = Circle(56, 224, wgt, wgt_list, 390)
        self.circle20 = Circle(56, 224, wgt, wgt_list, 408)
        self.circle21 = Circle(56, 224, wgt, wgt_list, 426)
        self.circle22 = Circle(56, 224, wgt, wgt_list, 444)
        self.circle23 = Circle(56, 224, wgt, wgt_list, 462)
        self.circle24 = Circle(56, 224, wgt, wgt_list, 480)
        self.circle25 = Circle(56, 224, wgt, wgt_list, 498)
        self.circle26 = Circle(56, 224, wgt, wgt_list, 516)
        self.circle27 = Circle(56, 224, wgt, wgt_list, 534)
        self.circle28 = Circle(56, 224, wgt, wgt_list, 552)
        self.circle29 = Circle(56, 224, wgt, wgt_list, 570)
        self.circle30 = Circle(56, 224, wgt, wgt_list, 588)
        self.circle31 = Circle(56, 224, wgt, wgt_list, 606)

        self.expc3 = ExpCircle(224, 144, 576, 2, wgt, wgt_list, 624)
        self.circle32 = Circle(184, 224, wgt, wgt_list, 648)
        self.expc4 = ExpCircle(576, 184, 736, 2, wgt, wgt_list, 666)
        self.circle33 = Circle(56, 224, wgt, wgt_list, 690)
        self.circle34 = Circle(56, 224, wgt, wgt_list, 708)
        self.circle35 = Circle(56, 224, wgt, wgt_list, 726)
        self.circle36 = Circle(56, 224, wgt, wgt_list, 744)
        self.circle37 = Circle(56, 224, wgt, wgt_list, 762)
        self.circle38 = Circle(56, 224, wgt, wgt_list, 780)
        self.circle39 = Circle(56, 224, wgt, wgt_list, 798)

        self.lateral1 = create_nobatSeqs(224, 128, 1, 1, 0, wgt, wgt_list, 816)
        self.lateral2 = create_nobatSeqs(576, 128, 1, 1, 0, wgt, wgt_list, 818)
        self.lateral3 = create_nobatSeqs(736, 128, 1, 1, 0, wgt, wgt_list, 820)
        self.fpn1 = create_nobatSeqs(128, 128, 3, 1, 1, wgt, wgt_list, 822)
        self.fpn2 = create_nobatSeqs(128, 128, 3, 1, 1, wgt, wgt_list, 824)
        self.fpn3 = create_nobatSeqs(128, 128, 3, 1, 1, wgt, wgt_list, 826)
        self.downsample1 = create_nobatSeqs(128, 128, 3, 2, 1, wgt, wgt_list, 828)
        self.downsample2 = create_nobatSeqs(128, 128, 3, 2, 1, wgt, wgt_list, 830)
        self.parfpn1 = create_nobatSeqs(128, 128, 3, 1, 1, wgt, wgt_list, 832)
        self.parfpn2 = create_nobatSeqs(128, 128, 3, 1, 1, wgt, wgt_list, 834)


        self.cls_stride1_1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.cls_stride1_1.weight = nn.parameter.Parameter(wgt[wgt_list[836]])
        self.gn1_1 = nn.GroupNorm(32, 256, 1e-5, True)
        self.gn1_1.weight = nn.parameter.Parameter(wgt[wgt_list[837]])
        self.gn1_1.bias = nn.parameter.Parameter(wgt[wgt_list[838]])

        self.cls_stride1_2 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.cls_stride1_2.weight = nn.parameter.Parameter(wgt[wgt_list[836]])
        self.gn1_2 = nn.GroupNorm(32, 256, 1e-5, True)
        self.gn1_2.weight = nn.parameter.Parameter(wgt[wgt_list[837]])
        self.gn1_2.bias = nn.parameter.Parameter(wgt[wgt_list[838]])

        self.cls_stride1_3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.cls_stride1_3.weight = nn.parameter.Parameter(wgt[wgt_list[836]])
        self.gn1_3 = nn.GroupNorm(32, 256, 1e-5, True)
        self.gn1_3.weight = nn.parameter.Parameter(wgt[wgt_list[837]])
        self.gn1_3.bias = nn.parameter.Parameter(wgt[wgt_list[838]])
        
        self.cls_stride2_1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.cls_stride2_1.weight = nn.parameter.Parameter(wgt[wgt_list[839]])
        self.gn2_1 = nn.GroupNorm(32, 256, 1e-5, True)
        self.gn2_1.weight = nn.parameter.Parameter(wgt[wgt_list[840]])
        self.gn2_1.bias = nn.parameter.Parameter(wgt[wgt_list[841]])

        self.cls_stride2_2 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.cls_stride2_2.weight = nn.parameter.Parameter(wgt[wgt_list[839]])
        self.gn2_2 = nn.GroupNorm(32, 256, 1e-5, True)
        self.gn2_2.weight = nn.parameter.Parameter(wgt[wgt_list[840]])
        self.gn2_2.bias = nn.parameter.Parameter(wgt[wgt_list[841]])

        self.cls_stride2_3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.cls_stride2_3.weight = nn.parameter.Parameter(wgt[wgt_list[839]])
        self.gn2_3 = nn.GroupNorm(32, 256, 1e-5, True)
        self.gn2_3.weight = nn.parameter.Parameter(wgt[wgt_list[840]])
        self.gn2_3.bias = nn.parameter.Parameter(wgt[wgt_list[841]])

        self.stride_cls1 = create_nobatSeqs(256, 2, 3, 1, 1, wgt, wgt_list, 842)
        self.stride_cls2 = create_nobatSeqs(256, 2, 3, 1, 1, wgt, wgt_list, 842)
        self.stride_cls3 = create_nobatSeqs(256, 2, 3, 1, 1, wgt, wgt_list, 842)
        self.stride_reg1 = create_nobatSeqs(256, 2, 3, 1, 1, wgt, wgt_list, 844)
        self.stride_reg2 = create_nobatSeqs(256, 2, 3, 1, 1, wgt, wgt_list, 844)
        self.stride_reg3 = create_nobatSeqs(256, 2, 3, 1, 1, wgt, wgt_list, 844)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()


    def forward(self, x, det_scale):
        x = self.relu(self.conv_0(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_4(x))
        x = self.maxpool_6(x)

        x = self.expc1(x)
        x = self.circle5(self.circle4(self.circle3(self.circle2(self.circle1(x)))))
        x = self.circle10(self.circle9(self.circle8(self.circle7(self.circle6(x)))))
        x = self.circle15(self.circle14(self.circle13(self.circle12(self.circle11(x)))))
        x = self.circle19(self.circle18(self.circle17(self.expc2(self.circle16(x)))))
        x = self.circle21(self.circle20(x))
        x = self.circle26(self.circle25(self.circle24(self.circle23(self.circle22(x)))))
        x1 = self.circle31(self.circle30(self.circle29(self.circle28(self.circle27(x)))))
        x2 = self.circle32(self.expc3(x1)) #
        x3 = self.circle36(self.circle35(self.circle34(self.circle33(self.expc4(x2)))))
        x3 = self.circle39(self.circle38(self.circle37(x3))) #

        x1 = self.lateral1(x1)
        x2 = self.lateral2(x2)
        x3 = self.lateral3(x3)
        m1 = self.up(x3) + x2
        up1 = self.up(m1)
        m2 = up1 + x1
        f1 = self.fpn1(m2)
        f2 = self.fpn2(m1)
        f3 = self.fpn3(x3)
        
        d1 = self.downsample1(f1)
        a1 = f2 + d1
        pf1 = self.parfpn1(a1)
        d2 = self.downsample2(a1)
        a2 = f3 + d2
        pf2 = self.parfpn2(a2)

        result1 = self.relu(self.gn1_1(self.cls_stride1_1(f1)))
        result1 = self.relu(self.gn2_1(self.cls_stride2_1(result1)))

        result2 = self.relu(self.gn1_2(self.cls_stride1_2(pf1)))
        result2 = self.relu(self.gn2_2(self.cls_stride2_2(result2)))

        result3 = self.gn1_3(self.cls_stride1_3(pf2))
        result3 = self.relu(result3)

        result3 = self.gn2_3(self.cls_stride2_3(result3))
        result3 = self.relu(result3)


        result1_cls = self.stride_cls1(result1)
        result1_reg = (self.stride_reg1(result1)) * 0.90731763
        result2_cls = self.stride_cls2(result2)
        result2_reg = (self.stride_reg2(result2)) * 1.41017079
        result3_cls = self.stride_cls3(result3)
        result3_reg = (self.stride_reg3(result3)) * 1.83822488

        result1_cls = result1_cls[0].detach()
        result1_reg = result1_reg[0].detach()
        result2_cls = result2_cls[0].detach()
        result2_reg = result2_reg[0].detach()
        result3_cls = result3_cls[0].detach()
        result3_reg = result3_reg[0].detach()

        result1_cls = result1_cls.permute(1, 2, 0).reshape(-1, 1).sigmoid()
        result1_reg = result1_reg.permute(1, 2, 0)
        result1_reg = result1_reg.reshape((-1,4)) * 8

        result2_cls = result2_cls.permute(1, 2, 0).reshape(-1, 1).sigmoid()
        result2_reg = result2_reg.permute(1, 2, 0)
        result2_reg = result2_reg.reshape((-1,4)) * 16

        result3_cls = result3_cls.permute(1, 2, 0).reshape(-1, 1).sigmoid()
        result3_reg = result3_reg.permute(1, 2, 0)
        result3_reg = result3_reg.reshape((-1,4)) * 32

        anchor1 = make_anchor(8, 640)
        box1 = distance2bbox(anchor1, result1_reg, max_shape=(360, 640, 3))

        anchor2 = make_anchor(16, 640)
        box2 = distance2bbox(anchor2, result2_reg, max_shape=(360, 640, 3))

        anchor3 = make_anchor(32, 640)
        box3 = distance2bbox(anchor3, result3_reg, max_shape=(360, 640, 3))
    
        boxes = torch.concat([box1, box2, box3])
        # boxes /= boxes.new_tensor([0.33333334, 0.33333334, 0.33333334, 0.33333334])
        boxes *= det_scale
        scores = torch.concat([result1_cls, result2_cls, result3_cls])
        
        # padding = scores.new_zeros(scores.shape[0], 1)
        # scores = torch.cat([scores, padding], dim=1)

        pos_inds = np.where(scores>=0.4)[0]
        pos_scores = scores[pos_inds]
        pos_bboxes = boxes[pos_inds]

        return pos_scores, pos_bboxes


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = torch.load("weights/scrfd.pth") # 가중치 경로
    wgt_list = list(weight['state_dict'].keys())
    model = Scrfd(weight['state_dict'], wgt_list)
    model.eval()
    # torch.save(model.state_dict(), "new_scrfd.pth")

    img = cv2.imread("../TestImage/people_test.jpg")
    input_size = [640, 640]

    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio>model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = img.shape[0] / float(new_height)
    resized_img = cv2.resize(img, (new_width, new_height))

    det_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img = det_img * 127
    det_img[:new_height, :new_width, :] = resized_img

    blob = cv2.dnn.blobFromImage(det_img, 1.0/128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
    blob = torch.Tensor(blob)
    with torch.no_grad():
        scores, bboxes = model(blob, det_scale)

    scores = np.array(scores)
    bboxes = np.array(bboxes)
    scores_ravel = scores.ravel()

    order = scores_ravel.argsort()[::-1]
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    det = pre_det[keep, :]

    for i in range(0, len(det)):
        box = det[i]
        color = (0, 0, 255)
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    
    while True:
        cv2.imshow("res", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break