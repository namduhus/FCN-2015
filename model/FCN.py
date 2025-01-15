## VGG16 Backbone model

###########import##################
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


###################################

##############FCN##################
class FCN(nn.Module):
    def __init__(self, num_classes=21, mode="32s"):
        super(FCN, self).__init__()
        self.mode = mode


        ## Pretrain VGG16
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children()) # VGG16의 Conv레이어 가져오기
        # features: Conv Layer + MaxPooling Layer(합성곱 기반 특징 추출 부분)
        # classifier: Fully Connected Layer(완전 연결 기반 분류 부분)
        # children(): Sequential 내에 포함된 각 레이어를 순서대로 반환하는 이터레이터 역활
        # VGG16은 13개의 합성곱 + 5개의 MaxPooling이 포함 -> list형태로 변환

        # Conv layers 가져오기 (pool5까지)
        # VGG16의 특정구간을 잘라서 할당
        self.conv1 = nn.Sequential(*features[:5])
        # Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d

        self.conv2 = nn.Sequential(*features[5:10])
        # Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d

        self.conv3 = nn.Sequential(*features[10:17])
        # Conv2d -> ReLU -> Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d

        self.conv4 = nn.Sequential(*features[17:24])
        # Conv2d -> ReLU -> Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d

        self.conv5 = nn.Sequential(*features[24:])
        # Conv2d -> ReLU -> Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d

        # Fully Convolutional Layers
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.score_fr = nn.Conv2d(4096, num_classes, 1)

        # 1x1 conv layers skip connections
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        # Transposed Layer
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, padding=8)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        # VGG16 Forward Pass
        x1 = self.conv1(x) #Feature Map 1
        x2 = self.conv2(x1) #Feature Map 2
        x3 = self.conv3(x2) #Feature Map 3
        x4 = self.conv4(x3) #Feature Map 4
        x5 = self.conv5(x4) #Feature Map 5



        # Fully Convolutional Layers
        x5 = self.relu(self.fc6(x5))
        x5 = self.dropout(x5)

        x5 = self.relu(self.fc7(x5))
        x5 = self.dropout(x5)

        score_fr = self.score_fr(x5)   # 최종 맵


        if self.mode == "16s":
            # FCN-16s: skip connection
            score_pool4 = self.score_pool4(x4)
            upscore2 = F.interpolate(score_fr, size=score_pool4.shape[2:], mode='bilinear', align_corners=False)
            fuse_pool4 = upscore2 + score_pool4  # Skip connection from pool4
            upscore16 = self.upscore16(fuse_pool4)  # Upsample to (224, 224)
            return F.interpolate(upscore16, size=x.shape[2:], mode='bilinear', align_corners=False)


        elif self.mode == "8s":
            # FCN-8s
            score_pool4 = self.score_pool4(x4)  # pool4: (14, 14)
            score_pool3 = self.score_pool3(x3)  # pool3: (28, 28)
            upscore2 = F.interpolate(score_fr, size=score_pool4.shape[2:], mode='bilinear', align_corners=False)
            fuse_pool4 = upscore2 + score_pool4  # Skip connection from pool4
            upscore2_pool4 = F.interpolate(fuse_pool4, size=score_pool3.shape[2:], mode='bilinear', align_corners=False)
            fuse_pool3 = upscore2_pool4 + score_pool3  # Skip connection from pool3
            upscore8 = self.upscore8(fuse_pool3)  # Upsample to (224, 224)
            return F.interpolate(upscore8, size=x.shape[2:], mode='bilinear', align_corners=False)

# # Test
# if __name__ == "__main__":
#     num_classes = 21
#     model = FCN(num_classes=num_classes, mode="8s").to("cuda" if torch.cuda.is_available() else "cpu")
#
#     input_tensor = torch.randn(1, 3, 224, 224).to("cuda" if torch.cuda.is_available() else "cpu")
#     output_tensor = model(input_tensor)
#
#     print(f"입력 크기: {input_tensor.shape}")
#     print(f"출력 크기: {output_tensor.shape}")