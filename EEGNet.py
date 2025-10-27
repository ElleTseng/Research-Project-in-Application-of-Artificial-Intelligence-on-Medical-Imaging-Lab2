import torch.nn as nn
import torch.nn.functional as F

# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self):    #定義網路的層結構（每一層長怎樣）
        super(EEGNet, self).__init__()
        # first conv: 提取時間特徵
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25), bias=False), #輸入有 1 個 channel; 輸出會有 16 個 feature maps; 只在時間維度做濾波; 步幅都為 1（不跳格）; 在高度方向 padding 0（不擴高），時間方向左右各補25; 因為 BN 本身有可學習的平移參數（beta），會取代 bias 的功能，所以關 bias
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        ) #(batch, 1, 2, 750)->(batch, 16, 2, 750)
        # depthwise conv: 提取空間(通道間)關聯性
        self.depthwiseConv = nn.Sequential(
            #(batch, 32, 1, 750)
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            #(batch, 32, 1, 187)
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(0.35)
        )
        # separable conv: 進一步萃取高階時間特徵
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            #再次做 pooling 和 dropout，進一步降低時間維度與防止過擬合
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0), #(batch, 32, 1, 23)
            nn.Dropout(0.35)
        ) 
        # classifier
        self.classify = nn.Sequential(
            nn.Flatten(), #把輸出結果攤平 #(batch, 736)
            nn.Linear(32*1*23, 2, bias=True)
        )

    def forward(self, x):  #定義資料如何通過這些層
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)  
        return x

# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self, n_output):
        super(DeepConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5),bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.45),

            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.45),

            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.45),

            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.45),
        )

        # Adaptive pooling → Flatten → Linear
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,20)),  # 高度=1, 寬度=20
            nn.Flatten(),
            nn.Linear(200*1*20, n_output)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x