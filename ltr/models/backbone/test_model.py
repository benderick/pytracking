from .timm_backbone import TimmBackboneWrapper
model = TimmBackboneWrapper('convnextv2_mbfd_femto', pretrained=False, output_layers=['0', '1', '2', '3'])

# import timm
# model = timm.create_model('ese_vovnet19b_slim_dw', pretrained=False, features_only=True)
import torch
x = torch.randn(1, 3, 512, 512)
y = model.forward(x)
for f, i in enumerate(y):
    print(f"Feature map {f}: shape {i.shape}")
