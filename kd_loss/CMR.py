import torch
import torch.nn as nn

def CMR(featureX,featureY):
    b,c,h,w=featureX.shape
    feature1=featureX.reshape(b,c,-1)
    feature2=featureY.reshape(b,h*w,-1)
    out=torch.bmm(feature2,feature1)
    out=torch.softmax(out,2)
    return out


if __name__ == '__main__':
    r1_1 = torch.randn(1, 320, 13 ,13)
    r1_2 = torch.randn(1, 320, 13 ,13)
    r2_1 = torch.randn(1, 320, 13 ,13)
    r2_2 = torch.randn(1, 320, 13 ,13)
    cmr1=CMR(r1_1,r1_2)
    cmr2=CMR(r2_1,r2_2)
    loss = nn.MSELoss(reduction='mean')
    print('loss:',loss(cmr1,cmr2))

