"""Pretrained CNN backbone + sex input + regression head."""
import timm
import torch
import torch.nn as nn

import config


class BoneAgeModel(nn.Module):
    def __init__(self, backbone=config.BACKBONE, pretrained=True):
        super().__init__()
        # num_classes=0 removes the ImageNet classifier; we keep the pooled features
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.sex_fc = nn.Sequential(nn.Linear(1, config.SEX_DIM), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features + config.SEX_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1),
        )

    def forward_from_features(self, feature_map, sex):
        """Pool the backbone feature map and predict (split out so Grad-CAM can reuse it)."""
        img_feat = self.backbone.forward_head(feature_map, pre_logits=True)
        return self.head(torch.cat([img_feat, self.sex_fc(sex)], dim=1))

    def forward(self, img, sex):
        return self.forward_from_features(self.backbone.forward_features(img), sex)


if __name__ == "__main__":
    model = BoneAgeModel()
    img = torch.randn(4, 3, config.IMG_SIZE, config.IMG_SIZE)
    sex = torch.randint(0, 2, (4, 1)).float()
    out = model(img, sex)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"backbone={config.BACKBONE}, params={n_params / 1e6:.1f}M, output shape={tuple(out.shape)}")
