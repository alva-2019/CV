import torch.nn as nn

'''Write for deeplabv3+，Total stride is 16倍
'''

class SepConvBNReLU(nn.Sequential):
    def __init__(self,input_chan,output_chan,s,p):
        super(SepConvBNReLU,self).__init__(
            nn.Conv2d(input_chan,input_chan,3,s,p,groups=input_chan,bias=False),
            nn.Conv2d(input_chan,output_chan,1,bias=False),
            nn.BatchNorm2d(output_chan),
            nn.ReLU(inplace=True))


#定义一般卷积的各个layers（含BN、BN\ReLU），组成一个Sequential
class ConvBNReLU(nn.Sequential):
    def __init__(self,input_chan,output_chan,k,s,p):
        super(ConvBNReLU,self).__init__(
            nn.Conv2d(input_chan,output_chan,k,s,p,bias=False),
            nn.BatchNorm2d(output_chan),
            nn.ReLU(inplace=True))

#定义Entry Flow的block
class EntryBlock(nn.Module):
    def __init__(self,input_chan,output_chan):
        super(EntryBlock,self).__init__()
        self.conv1=SepConvBNReLU(input_chan, output_chan,1,1)
        self.conv2=SepConvBNReLU(output_chan,output_chan,1,1)
        self.conv3=SepConvBNReLU(output_chan,output_chan,2,1)

        self.skip=ConvBNReLU(input_chan,output_chan,1,2,0)

    def forward(self, x):
        identity=self.skip(x)
        out=self.conv3(self.conv2(self.conv1(x)))
        out +=identity
        return out

#Middle Flow的block
class MiddleBlock(nn.Module):
    def __init__(self,input_chan):
        super(MiddleBlock,self).__init__()
        self.conv1=SepConvBNReLU(input_chan,input_chan,1,1)
        self.conv2=SepConvBNReLU(input_chan,input_chan,1,1)
        self.conv3=SepConvBNReLU(input_chan,input_chan,1,1)

    def forward(self, x):
        identity=x
        out=self.conv3(self.conv2(self.conv1(x)))
        out +=identity
        return out


#定义Xception 的backbone
class Xception(nn.Module):
    def __init__(self):    #非分类问题，初始化不用输入分类参数num_classes
        super(Xception, self).__init__()
        MidLayers = []

        # Entry Flow
        self.block1 = ConvBNReLU(3,32,3,2,0)
        self.block2 = ConvBNReLU(32,64,3,1,1)
        self.block3 = EntryBlock(64,128)
        self.block4 = EntryBlock(128,256)
        self.block5 = EntryBlock(256,728)

        # Middle Flow
        for i in range(16):
            MidLayers +=[MiddleBlock(728)]
        self.Middle=nn.Sequential(*MidLayers)

   

    def forward(self,x):
        #Entry Flow
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        shortcut=x
        x=self.block4(x)
        x=self.block5(x)

        #Middle Flow
        IdentifyOne=x
        MData=self.Middle(x)
        MData +=IdentifyOne
        '''
        #Exit Flow
        identify2=self.skip2(MData)
        y=self.ExitStage(MData)
        y +=identify2
        data=self.block6(y)'''
        return shortcut,MData








