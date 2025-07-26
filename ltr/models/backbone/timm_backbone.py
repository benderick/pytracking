import torch.nn as nn
class TimmBackboneWrapper(nn.Module):
    """Timm模型的包装器，使其与框架兼容"""
    def __init__(self, model_name, pretrained=True, output_layers=None):
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.output_layers = output_layers if output_layers else ['0', '1', '2', '3']
        
    def forward(self, x, layers=None):
        # 如果layers为None，使用默认层
        if layers is None:
            layers = self.output_layers
            
        # 使用timm的features_only模式获取中间特征
        features = self.model(x)
        
        # 将特征映射到字典格式，与其他backbone兼容
        feature_dict = {}
        for i, feat in enumerate(features):
            layer_name = str(i)
            feature_dict[layer_name] = feat
            
        # 返回请求的层
        return {layer: feature_dict[layer] for layer in layers if layer in feature_dict}
    
if __name__ == "__main__":
    model = TimmBackboneWrapper('convnextv2_mbfd_femto', pretrained=False, output_layers=['0', '1', '2', '3'])

    # import timm
    # model = timm.create_model('ese_vovnet19b_slim_dw', pretrained=False, features_only=True)
    import torch
    x = torch.randn(1, 3, 512, 512)
    y = model.forward(x)
    for k, v in y.items():
        print(f"Feature map {k}: shape {v.shape}")
