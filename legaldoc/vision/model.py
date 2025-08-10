import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional

class DocumentCNN(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = 'resnet50', pretrained: bool = True):
        super(DocumentCNN, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classify
        output = self.classifier(attended_features)
        
        return output, attention_weights

class MultiScaleDocumentCNN(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(MultiScaleDocumentCNN, self).__init__()
        
        # Multiple scale branches
        self.scale1 = models.resnet18(pretrained=pretrained)
        self.scale2 = models.resnet18(pretrained=pretrained)
        self.scale3 = models.resnet18(pretrained=pretrained)
        
        # Remove final layers
        feature_dim = self.scale1.fc.in_features
        self.scale1.fc = nn.Identity()
        self.scale2.fc = nn.Identity()
        self.scale3.fc = nn.Identity()
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-scale processing
        # Scale 1: Full resolution
        scale1_features = self.scale1(x)
        
        # Scale 2: Half resolution
        x_scale2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        scale2_features = self.scale2(x_scale2)
        
        # Scale 3: Quarter resolution
        x_scale3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        scale3_features = self.scale3(x_scale3)
        
        # Concatenate features
        combined_features = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)
        
        # Fusion and classification
        fused_features = self.fusion(combined_features)
        output = self.classifier(fused_features)
        
        return output

class DocumentTransformer(nn.Module):
    def __init__(self, num_classes: int = 2, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super(DocumentTransformer, self).__init__()
        
        # CNN feature extractor
        self.cnn_backbone = models.resnet18(pretrained=True)
        cnn_feature_dim = self.cnn_backbone.fc.in_features
        self.cnn_backbone.fc = nn.Identity()
        
        # Project CNN features to transformer dimension
        self.feature_projection = nn.Linear(cnn_feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 197, d_model))  # 14x14 + 1 cls token
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract CNN features
        cnn_features = self.cnn_backbone(x)  # [batch_size, cnn_feature_dim]
        
        # Project to transformer dimension
        features = self.feature_projection(cnn_features)  # [batch_size, d_model]
        features = features.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)  # [batch_size, 2, d_model]
        
        # Add positional encoding
        features += self.pos_encoding[:, :features.size(1), :]
        
        # Transformer processing
        transformer_output = self.transformer(features)
        
        # Use CLS token for classification
        cls_output = transformer_output[:, 0, :]  # [batch_size, d_model]
        
        # Classification
        output = self.classifier(cls_output)
        
        return output

def create_model(model_type: str = 'cnn', num_classes: int = 2, **kwargs) -> nn.Module:
    """Factory function to create different model types"""
    
    if model_type == 'cnn':
        backbone = kwargs.get('backbone', 'resnet50')
        pretrained = kwargs.get('pretrained', True)
        return DocumentCNN(num_classes, backbone, pretrained)
    
    elif model_type == 'multiscale_cnn':
        pretrained = kwargs.get('pretrained', True)
        return MultiScaleDocumentCNN(num_classes, pretrained)
    
    elif model_type == 'transformer':
        d_model = kwargs.get('d_model', 512)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 6)
        return DocumentTransformer(num_classes, d_model, nhead, num_layers)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(model: nn.Module):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
