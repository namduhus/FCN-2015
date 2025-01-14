## VGG Backbone model

###########import##################
import torch
import torch.nn as nn
import torchvision.models as models
###################################

##############FCN##################
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()


        ## Pretrain VGG16
        vgg = models.vgg16(pretrained=True)
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

        # Fully Convolutional Layers (Transposed / Up-Sampling)
        self.upconv1 = nn.ConvTranspose2d(512,
                                          256,
                                          4,
                                          2,
                                          1) # 2x UpSampling
        self.upconv2 = nn.ConvTranspose2d(256,
                                          128,
                                          4,
                                          2,
                                          1) # 2x UpSampling
        self.upconv3 = nn.ConvTranspose2d(128,
                                          num_classes,
                                          16,
                                          8,
                                          4) # 8x UpSampling


    def forward(self, x):
        x1 = self.conv1(x) #Feature Map 1
        x2 = self.conv2(x1) #Feature Map 2
        x3 = self.conv3(x2) #Feature Map 3
        x4 = self.conv4(x3) #Feature Map 4
        x5 = self.conv5(x4) #Feature Map 5

        # UpSampling
        x = self.upconv1(x5)  # 1nd UpSampling
        x = self.upconv2(x)   # 2nd UpSampling
        x = self.upconv3(x)   # last UpSampling
        return x
# Test
# num_classes = 21
# model = FCN(num_classes=num_classes)
#
# device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# input_tensor = torch.randn(1, 3, 224, 224).to(device)
#
# output_tensor = model(input_tensor)
#
# print(f"입력 크기: {input_tensor.shape}")
# print(f"출력 크기: {output_tensor.shape}")