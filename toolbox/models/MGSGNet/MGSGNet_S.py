import torch
import torch.nn as nn
from toolbox.models.MGSGNet.segformer.mix_transformer import mit_b0
from torch.nn import functional as F
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x=self.relu(x)
        return x
        
class CAFM(nn.Module):
    def __init__(self,channel):
        super(CAFM, self).__init__()
        self.channel=channel
        self.Rmlp=nn.Conv2d(self.channel,1,1,1,0)
        self.Dmlp = nn.Conv2d(self.channel, 1, 1, 1, 0)
        self.Rpa=nn.Linear(26*26,self.channel)
        self.Dpa = nn.Linear(26*26,self.channel)
    def forward(self, rgbFeature,depthFeature):
        b,c,h,w=rgbFeature.shape

        thea=self.Rmlp(rgbFeature)
        beta = self.Dmlp(depthFeature)
        thea=F.interpolate(thea,size=(26, 26), mode='bilinear')
        beta = F.interpolate(beta, size=(26, 26), mode='bilinear')
        rgbthea=nn.Tanh()(thea+beta)
        depthbeta=nn.Tanh()(thea+beta)
        rgbM=torch.matmul(thea.reshape(b,1,26*26).permute(0,2,1),rgbthea.reshape(b,1,26*26))
        depthM = torch.matmul(beta.reshape(b, 1, 26 * 26).permute(0, 2, 1), depthbeta.reshape(b, 1, 26 * 26))
        rgbM=self.Rpa(rgbM).reshape(b,26*26,self.channel).permute(0,2,1).reshape(b,self.channel,26,26)
        depthM=self.Dpa(depthM).reshape(b,26*26,self.channel).permute(0,2,1).reshape(b,self.channel,26,26)
        rgbM = F.interpolate(rgbM, size=(h, w), mode='bilinear')
        depthM = F.interpolate(depthM, size=(h, w), mode='bilinear')
        rgbM=rgbM*rgbFeature+rgbFeature
        depthM=depthM*depthFeature+depthFeature
        fused=nn.Sigmoid()(rgbM*depthM)*rgbFeature+rgbFeature
        return fused

class NLC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(NLC, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class EGCN(nn.Module):
    def __init__(self, in_channels=256,in_channel=32, out_channel=32):
        super(EGCN, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel*2, out_channel, 3,padding=1),
            BasicConv2d(in_channel, out_channel, 3,padding=1)
        )
        self.conv3_1=nn.Conv2d(in_channels,in_channel,1)
        self.conv3_2 = nn.Conv2d(in_channel, in_channel, 3,padding=1)

    def forward(self, x4,xr,xb):
        BiggerSize=xb.size()
        k=F.interpolate(self.conv3_1(x4), size=BiggerSize[2:], mode='bilinear', align_corners=True)
        v=self.conv3_2(xr)
        q=nn.Sigmoid()(v)
        xr_3=q*k+v
        out=self.branch0(torch.cat([xr_3,xb],dim=1))+xr
        return out


class SGCN(nn.Module):
    def __init__(self, plane):
        super(SGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return nn.Sigmoid()(out)


class DGCN(nn.Module):
    def __init__(self, planes, ratio=4):
        super(DGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return nn.Sigmoid()(out)


class BoundAware(nn.Module):
    def __init__(self, inplane, skip_num, norm_layer):
        super(BoundAware, self).__init__()
        self.reduceDim = skip_num
        self.pre_extractor = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3,
                      padding=1, groups=1, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.extractor = nn.Sequential(
            nn.Conv2d(inplane + skip_num, inplane, kernel_size=3,
                      padding=1, groups=8, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

    def forward(self, aspp, layer1):
        aspp_up=F.interpolate(aspp, size=layer1.size()[2:], mode='bilinear',align_corners=True)
        seg_edge = torch.cat([aspp_up, layer1], dim=1)
        seg_edge = self.extractor(seg_edge)
        seg_body = aspp_up - seg_edge

        return seg_edge, seg_body


class SGM_s(nn.Module):

    def __init__(self, channel,inchannels = [256, 160, 64, 32]):
        super(SGM_s, self).__init__()
        self.guideLayers=4
        self.reduceDim=48
        self.reduceBot = nn.Conv2d(32, self.reduceDim, kernel_size=1, bias=False)
        self.reduceBots = nn.ModuleList()
        for i in range(self.guideLayers):
            self.reduceBots.append(nn.Conv2d(inchannels[i], 48, kernel_size=1, bias=False))
            
        self.binary_fuse = [nn.Conv2d(channel + 48, channel, kernel_size=1, bias=False) for _ in range(self.guideLayers)]
        self.binary_fuse = nn.ModuleList(self.binary_fuse)


        self.HR = nn.ModuleList([nn.Conv2d(channel, 48, kernel_size=1, bias=False)
                                       for i in range(self.guideLayers)])

        self.reduceBotAsp = nn.Conv2d(256, channel, kernel_size=1, bias=False)

        self.boundAwares = [BoundAware(channel, norm_layer=nn.BatchNorm2d, skip_num=48)
                                for _ in range(self.guideLayers)]
        self.boundAwares = nn.ModuleList(self.boundAwares)

        self.gAwares = [ EGCN(in_channels=inchannels[i]) for i in range(self.guideLayers)]
        self.gAwares = nn.ModuleList(self.gAwares)

        self.bound_out_pre = [nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)) for _ in range(self.guideLayers)]
        self.bound_out_pre = nn.ModuleList(self.bound_out_pre)
        self.bound_out = nn.ModuleList([nn.Conv2d(channel, 1, kernel_size=1, bias=False)
                                       for _ in range(self.guideLayers)])

        self.bound_out_ff = nn.ModuleList([nn.Conv2d(channel, 2, kernel_size=1, bias=False)
                                          for _ in range(self.guideLayers)])

        self.binary_out_pre = [nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)) for _ in range(self.guideLayers)]
        self.binary_out_pre = nn.ModuleList(self.binary_out_pre)
        self.binary_out = nn.ModuleList([nn.Conv2d(channel, 2, kernel_size=1, bias=False)
                                       for _ in range(self.guideLayers)])

        self.semantic_out_pre = [nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)) for _ in range(self.guideLayers - 1)]
        self.semantic_out_pre.append(nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)))
        self.semantic_out_pre = nn.ModuleList(self.semantic_out_pre)
        self.semantic_out = nn.ModuleList([nn.Conv2d(channel, 9, kernel_size=1, bias=False)
                                            for _ in range(self.guideLayers)])

    def forward(self, xin, x5,x4,x3,x2):
        outputSize=(480 ,640)
        BiggerSize=x2.size()
        allEncode=[x5,x4,x3,x2]
        seg_bounds = []
        seg_bound_outs = []
        seg_binarys = []
        seg_binary_outs = []
        seg_Semantics = []
        seg_Semantics_outs = []
        aspp = self.reduceBotAsp(xin)
        final_fuse_feat = F.interpolate(aspp, size=BiggerSize[2:], mode='bilinear', align_corners=True)
        low_feat = self.reduceBot(F.interpolate(aspp, size=(120, 160), mode='bilinear', align_corners=True)*x2)

        for i in range(self.guideLayers):
            if i == 0:
                last_seg_feat = aspp
            else:
                last_seg_feat = seg_Semantics[-1]
                last_seg_feat = F.interpolate(last_seg_feat, size=aspp.size()[2:],
                                              mode='bilinear', align_corners=True)  

            seg_edge, seg_body = self.boundAwares[i](last_seg_feat, low_feat)  


            high_fine = self.HR[i](seg_body)*F.interpolate(self.reduceBots[i](allEncode[i]), size=BiggerSize[2:], mode='bilinear',
                                      align_corners=True)
            seg_body = self.binary_fuse[i](torch.cat([seg_body, high_fine], dim=1))  
            seg_body_pre = self.binary_out_pre[i](seg_body)
            seg_binary_out = F.interpolate(self.binary_out[i](seg_body_pre), size=outputSize,
                                         mode='bilinear', align_corners=True)  
            seg_binarys.append(seg_body_pre)
            seg_binary_outs.append(nn.Sigmoid()(seg_binary_out))

            seg_edge_pre = self.bound_out_pre[i](seg_edge)  

            seg_bound_out_pre1 = self.bound_out_ff[i](seg_edge_pre)  
            seg_bound_out = F.interpolate(seg_bound_out_pre1, size=outputSize,
                                         mode='bilinear', align_corners=True)  
            seg_bounds.append(seg_edge_pre)
            seg_bound_outs.append(nn.Sigmoid()(seg_bound_out))

            seg_out = seg_body + seg_edge
            seg_out = self.gAwares[i](allEncode[i],seg_out,seg_edge_pre)

            if i >= self.guideLayers - 1:
                seg_final_pre = self.semantic_out_pre[i](torch.cat([final_fuse_feat, seg_out], dim=1))
            else:
                seg_final_pre = self.semantic_out_pre[i](seg_out)
            seg_final_out = F.interpolate(self.semantic_out[i](seg_final_pre), size=outputSize,
                                          mode='bilinear', align_corners=True)
            seg_Semantics.append(seg_final_pre)
            seg_Semantics_outs.append(nn.Sigmoid()(seg_final_out))

        return seg_Semantics_outs, seg_binary_outs, seg_bound_outs


class MGSGNet_S(nn.Module):
    def __init__(self, channel=32):
        super(MGSGNet_S, self).__init__()
        # Backbone
        self.rgb = mit_b0()
        self.rgb.init_weights("/home/wby/Desktop/MGSGNet/toolbox/models/MGSGNet/segformer/pretrained/mit_b0.pth")
        self.depth = mit_b0()
        self.depth.init_weights("/home/wby/Desktop/MGSGNet/toolbox/models/MGSGNet/segformer/pretrained/mit_b0.pth")
        self.cafm1=CAFM(32)
        self.cafm2 = CAFM(64)
        self.cafm3 = CAFM(160)
        self.cafm4 = CAFM(256)
        # Decoder
        self.nlc = NLC(256,256)
        self.sgm = SGM_s(channel)

    def forward(self, x, x_depth):

        x = self.rgb.forward_features(x)
        x_depth = self.depth.forward_features(x_depth)
        #stage1
        x1 = x[0]
        x1_depth = x_depth[0]
        #stage2
        x2 = x[1]
        x2_depth = x_depth[1]
        #stage3
        x3_1 = x[2]
        x3_1_depth = x_depth[2]
        #stage4
        x4_1 = x[3]
        x4_1_depth = x_depth[3]
        #decoder
        x1_1 = self.cafm1(x1, x1_depth)
        x2_1 = self.cafm2(x2, x2_depth)
        x3_1 = self.cafm3(x3_1, x3_1_depth)
        x4_1 = self.cafm4(x4_1, x4_1_depth)
        x4_2 = self.nlc(x4_1)
        y = self.sgm(x4_2,x4_1,x3_1,x2_1, x1_1)
        return y


if __name__ == '__main__':
    img = torch.randn(1, 3, 480, 640).cuda()
    depth = torch.randn(1, 3, 480, 640).cuda()
    model = MGSGNet_S().to(torch.device("cuda:0"))
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    import numpy as np
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    out = model(img, depth)
    for i in range(len(out[0])):
        print(out[0][i].shape)
