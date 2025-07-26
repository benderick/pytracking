from backbone import resnet50
model = resnet50(pretrained=False, frozen_layers=())
import torch
x = torch.randn(1, 3, 512, 512)
outputs = model(x, ['conv1', 'layer1', 'layer2','layer3','layer4'])
for k, v in outputs.items():
        print(f"{k}: {v.shape}")
print(model._out_feature_channels)
print(model._out_feature_strides)
