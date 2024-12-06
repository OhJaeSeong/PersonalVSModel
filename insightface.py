import torch
import torch.nn as nn
import numpy as np
from onnx import numpy_helper

import torch.nn.functional as F

def create_seqs(inp, oup, kernel_size, stride, pad):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=True)
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def create_seqs_no_relu(inp, oup, kernel_size, stride, pad):
    conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=True)
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(oup),
    )

def fpn(input1, input2, input3):
    input3_up = F.interpolate(input3, size=[input2.size(2), input2.size(3)], mode="nearest")
    input2 = input2 + input3_up

    input2_up = F.interpolate(input2, size=[input1.size(2), input1.size(3)], mode="nearest")
    input1 = input1 + input2_up

    return input1, input2, input3

class Insightface(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Conv_0 = create_seqs(3, 28, 3, 2, 1)
        self.Conv_3 = create_seqs(28, 28, 3, 1, 1)
        self.Conv_6 = create_seqs(28, 56, 3, 1, 1)

        self.MaxPool_9 = nn.MaxPool2d(kernel_size=2, ceil_mode=False, stride=2, padding=0)

        self.Conv_10 = create_seqs(56, 56, 3, 1, 1)
        self.Conv_13 = create_seqs_no_relu(56, 56, 3, 1, 1)
        self.Relu_16 = nn.ReLU(inplace=True)

        self.Conv_17 = create_seqs(56, 56, 3, 1, 1)
        self.Conv_20 = create_seqs_no_relu(56, 56, 3, 1, 1)
        self.Relu_23 = nn.ReLU(inplace=True)

        self.Conv_24 = create_seqs(56, 56, 3, 1, 1)
        self.Conv_27 = create_seqs_no_relu(56, 56, 3, 1, 1)
        self.Relu_30 = nn.ReLU(inplace=True)

        self.Conv_31 = create_seqs(56, 88, 3, 2, 1)
        self.Conv_34 = create_seqs_no_relu(88, 88, 3, 1, 1)

        self.AveragePool_36 = nn.AvgPool2d(kernel_size=2, ceil_mode=True, stride=2, padding=0)
        self.Conv_37 = create_seqs_no_relu(56, 88, 1, 1, 0)
        self.Relu_40 = nn.ReLU(inplace=True)

        self.Conv_41 = create_seqs(88, 88, 3, 1, 1)
        self.Conv_44 = create_seqs_no_relu(88, 88, 3, 1, 1)
        self.Relu_47 = nn.ReLU(inplace=True)

        self.Conv_48 = create_seqs(88, 88, 3, 1, 1)
        self.Conv_51 = create_seqs_no_relu(88, 88, 3, 1, 1)
        self.Relu_54 = nn.ReLU(inplace=True)

        self.Conv_55 = create_seqs(88, 88, 3, 1, 1)
        self.Conv_58 = create_seqs_no_relu(88, 88, 3, 1, 1)
        self.Relu_61 = nn.ReLU(inplace=True) # 1차 분기

        self.Conv_62 = create_seqs(88, 88, 3, 2, 1)
        self.Conv_65= create_seqs_no_relu(88, 88, 3, 1, 1)
        
        self.AveragePool_67 = nn.AvgPool2d(kernel_size=2, ceil_mode=True, stride=2, padding=0)
        self.Conv_68 = create_seqs_no_relu(88, 88, 1, 1, 0)
        self.Relu_71 = nn.ReLU(inplace=True)

        self.Conv_72 = create_seqs(88, 88, 3, 1, 1)
        self.Conv_75 = create_seqs_no_relu(88, 88, 3, 1, 1)
        self.Relu_78 = nn.ReLU(inplace=True) # 2차 분기

        self.Conv_79 = create_seqs(88, 224, 3, 2, 1)
        self.Conv_82 = create_seqs_no_relu(224, 224, 3, 1, 1)
        
        self.AveragePool_84 = nn.AvgPool2d(kernel_size=2, ceil_mode=True, stride=2, padding=0)
        self.Conv_85 = create_seqs_no_relu(88, 224, 1, 1, 0)
        self.Relu_88 = nn.ReLU(inplace=True)

        self.Conv_89 = create_seqs(224, 224, 3, 1, 1)
        self.Conv_92 = create_seqs_no_relu(224, 224, 3, 1, 1)
        self.Relu_95 = nn.ReLU(inplace=True)

        self.Conv_96 = create_seqs(224, 224, 3, 1, 1)
        self.Conv_99 = create_seqs_no_relu(224, 224, 3, 1, 1)
        self.Relu_102 = nn.ReLU(inplace=True)
        
        self.Conv_103 = create_seqs_no_relu(88, 56, 1, 1, 0)
        self.Conv_104 = create_seqs_no_relu(88, 56, 1, 1, 0)
        self.Conv_105 = create_seqs_no_relu(224, 56, 1, 1, 0)

        self.Conv_146 = create_seqs_no_relu(56, 56, 3, 1, 1)
        self.Conv_147 = create_seqs_no_relu(56, 56, 3, 1, 1)
        self.Conv_148 = create_seqs_no_relu(56, 56, 3, 1, 1)
        self.Conv_149 = create_seqs_no_relu(56, 56, 3, 2, 1)

        self.Conv_151 = create_seqs_no_relu(56, 56, 3, 2, 1)
        self.Conv_153 = create_seqs_no_relu(56, 56, 3, 1, 1)
        self.Conv_154 = create_seqs_no_relu(56, 56, 3, 1, 1)
        
        self.Conv_155 = create_seqs(56, 80, 3, 1, 1)
        self.Conv_158 = create_seqs(80, 80, 3, 1, 1)
        self.Conv_161 = create_seqs(80, 80, 3, 1, 1)

        self.Conv_164 = create_seqs_no_relu(80, 2, 3, 1, 1)
        self.Conv_165 = create_seqs_no_relu(80, 8, 3, 1, 1)
        self.Conv_167 = create_seqs_no_relu(80, 20, 3, 1, 1)

        self.Conv_178 = create_seqs(56, 80, 3, 1, 1)
        self.Conv_181 = create_seqs(80, 80, 3, 1, 1)
        self.Conv_184 = create_seqs(80, 80, 3, 1, 1)

        self.Conv_187 = create_seqs_no_relu(80, 2, 3, 1, 1)
        self.Conv_188 = create_seqs_no_relu(80, 8, 3, 1, 1)
        self.Conv_190 = create_seqs_no_relu(80, 20, 3, 1, 1)

        self.Conv_201 = create_seqs(56, 80, 3, 1, 1)
        self.Conv_204 = create_seqs(80, 80, 3, 1, 1)
        self.Conv_207 = create_seqs(80, 80, 3, 1, 1)

        self.Conv_210 = create_seqs_no_relu(80, 2, 3, 1, 1)
        self.Conv_211 = create_seqs_no_relu(80, 8, 3, 1, 1)
        self.Conv_213 = create_seqs_no_relu(80, 20, 3, 1, 1)

        self.Mul_166 = torch.Tensor([0.8464])
        self.Mul_189 = torch.Tensor([0.8996])
        self.Mul_212 = torch.Tensor([1.0812])

        self.Reshape1 = [-1, 1]
        self.Reshape2 = [-1, 4]
        self.Reshape3 = [-1, 10]

        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.Conv_0(x)
        # print(torch.min(x1), torch.max(x1))
        # print(torch.min(self.Conv_201.state_dict()['0.weight']), torch.max(self.Conv_201.state_dict()['0.bias']))
        x1 = self.Conv_3(x1)
        x1 = self.Conv_6(x1)

        x1 = self.MaxPool_9(x1)

        x2 = self.Conv_10(x1)
        x2 = self.Conv_13(x2)
        x1 = x1 + x2 # Add_15
        x1 = self.Relu_16(x1)

        x2 = self.Conv_17(x1)
        x2 = self.Conv_20(x2)
        x1 = x1 + x2 # Add_22
        x1 = self.Relu_23(x1)

        x2 = self.Conv_24(x1)
        x2 = self.Conv_27(x2)
        x1 = x1 + x2 # ADD_29
        x1 = self.Relu_30(x1)

        x2 = self.AveragePool_36(x1)
        x2 = self.Conv_37(x2)
        x1 = self.Conv_31(x1)
        x1 = self.Conv_34(x1)
        x1 = x1 + x2
        x1 = self.Relu_40(x1)

        x2 = self.Conv_41(x1)
        x2 = self.Conv_44(x2)
        x1 = x1 + x2 # ADD_46
        x1 = self.Relu_47(x1)

        x2 = self.Conv_48(x1)
        x2 = self.Conv_51(x2)
        x1 = x1 + x2 # ADD_53
        x1 = self.Relu_54(x1)

        x2 = self.Conv_55(x1)
        x2 = self.Conv_58(x2)
        x1 = x1 + x2 # ADD_60
        x1 = self.Relu_61(x1)

        x2 = self.Conv_62(x1)
        x2 = self.Conv_65(x2)
        x3 = self.AveragePool_67(x1)
        x3 = self.Conv_68(x3)
        x2 = x2 + x3 # ADD_70
        x2 = self.Relu_71(x2)

        x3 = self.Conv_72(x2)
        x3 = self.Conv_75(x3)
        x2 = x2 + x3 # ADD_77
        x2 = self.Relu_78(x2)

        x3 = self.Conv_79(x2)
        x3 = self.Conv_82(x3)
        x4 = self.AveragePool_84(x2)
        x4 = self.Conv_85(x4)
        x3 = x3 + x4 # ADD_87
        x3 = self.Relu_88(x3)

        x4 = self.Conv_89(x3)
        x4 = self.Conv_92(x4)
        x3 = x3 + x4 # ADD_94
        x3 = self.Relu_95(x3)

        x4 = self.Conv_96(x3)
        x4 = self.Conv_99(x4)
        x3 = x3 + x4 # ADD_101
        x3 = self.Relu_102(x3)

        x1 = self.Conv_103(x1)
        x2 = self.Conv_104(x2)
        x3 = self.Conv_105(x3)

        x1, x2, x3 = fpn(x1, x2, x3)

        x1 = self.Conv_146(x1)
        x2 = self.Conv_147(x2)
        x3 = self.Conv_148(x3)

        x2_add = self.Conv_149(x1)
        x2 = x2 + x2_add # ADD_150

        x3_add = self.Conv_151(x2)
        x3 = x3 + x3_add # ADD_152

        # x1
        x1 = self.Conv_155(x1)
        x1 = self.Conv_158(x1)
        x1 = self.Conv_161(x1)

        x1_1 = self.Conv_164(x1)
        x1_1 = x1_1.permute(2, 3, 0, 1)
        x1_1 = torch.reshape(x1_1, (self.Reshape1[0], self.Reshape1[1]))
        x1_1 = self.sig(x1_1)

        x1_2 = self.Conv_165(x1)
        x1_2 = x1_2 * self.Mul_166
        x1_2 = x1_2.permute(2, 3, 0, 1)
        x1_2 = torch.reshape(x1_2, (self.Reshape2[0], self.Reshape2[1]))

        x1_3 = self.Conv_167(x1)
        x1_3 = x1_3.permute(2, 3, 0, 1)
        x1_3 = torch.reshape(x1_3, (self.Reshape3[0], self.Reshape3[1]))

        #x2
        x2 = self.Conv_153(x2)
        x2 = self.Conv_178(x2)
        x2 = self.Conv_181(x2)
        x2 = self.Conv_184(x2)

        x2_1 = self.Conv_187(x2)
        x2_1 = x2_1.permute(2, 3, 0, 1)
        x2_1 = torch.reshape(x2_1, (self.Reshape1[0], self.Reshape1[1]))
        x2_1 = self.sig(x2_1)

        x2_2 = self.Conv_188(x2)
        x2_2 = x2_2 * self.Mul_189
        x2_2 = x2_2.permute(2, 3, 0, 1)
        x2_2 = torch.reshape(x2_2, (self.Reshape2[0], self.Reshape2[1]))

        x2_3 = self.Conv_190(x2)
        x2_3 = x2_3.permute(2, 3, 0, 1)
        x2_3 = torch.reshape(x2_3, (self.Reshape3[0], self.Reshape3[1]))

        #x3
        x3 = self.Conv_154(x3)
        x3 = self.Conv_201(x3)
        x3 = self.Conv_204(x3)
        x3 = self.Conv_207(x3)

        x3_1 = self.Conv_210(x3)
        x3_1 = x3_1.permute(2, 3, 0, 1)
        x3_1 = torch.reshape(x3_1, (self.Reshape1[0], self.Reshape1[1]))
        x3_1 = self.sig(x3_1)

        x3_2 = self.Conv_211(x3)
        x3_2 = x3_2 * self.Mul_212
        x3_2 = x3_2.permute(2, 3, 0, 1)
        x3_2 = torch.reshape(x3_2, (self.Reshape2[0], self.Reshape2[1]))

        x3_3 = self.Conv_213(x3)
        x3_3 = x3_3.permute(2, 3, 0, 1)
        x3_3 = torch.reshape(x3_3, (self.Reshape3[0], self.Reshape3[1]))
        
        x1_1 = x1_1.detach().numpy(); x1_2 = x1_2.detach().numpy(); x1_3 = x1_3.detach().numpy()
        x2_1 = x2_1.detach().numpy(); x2_2 = x2_2.detach().numpy(); x2_3 = x2_3.detach().numpy()
        x3_1 = x3_1.detach().numpy(); x3_2 = x3_2.detach().numpy(); x3_3 = x3_3.detach().numpy()

        return [x1_1 ,x2_1 ,x3_1 ,x1_2, x2_2, x3_2, x1_3, x2_3, x3_3]