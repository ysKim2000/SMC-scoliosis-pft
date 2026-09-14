import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# Tabular encoders 
class TabularMLP(nn.Module):
    def __init__(self, n_in, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = n_in
        dropout = 0.1
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)

class TabularCNN1D(nn.Module):
    """[B, F] -> [B, 1, F] -> Conv1d -> GAP -> FC(128).

    Length-independent, so the parameter count stays fixed as the number of
    tabular features changes.
    """
    def __init__(self, n_in, channels=(16, 32), kernel=3, dropout=0.3):
        super().__init__()
        mods = []
        c_in = 1
        for c in channels:
            mods += [
                nn.Conv1d(c_in, c, kernel, padding=kernel//2, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ]
            c_in = c
        self.conv = nn.Sequential(*mods)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),                    # [B,C,1] -> [B,C]
            nn.Linear(c_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = 128

    def forward(self, x):
        x = x.unsqueeze(1)        # [B,1,F]
        x = self.conv(x)          # [B,C,F]
        x = self.gap(x)           # [B,C,1]
        x = self.fc(x)            # [B,128]
        return x

# Simple attention fusion examples 
class GeneralAttentionFusion(nn.Module):
    """Scalar gate over the two modalities, with a residual connection."""
    def __init__(self, d_img, d_tab, d_fuse=None, use_softmax=True, dropout=0.3):
        super().__init__()
        d_fuse = d_fuse or max(d_img, d_tab)
        self.proj_img = nn.Identity() if d_img == d_fuse else nn.Linear(d_img, d_fuse, bias=False)
        self.proj_tab = nn.Identity() if d_tab == d_fuse else nn.Linear(d_tab, d_fuse, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_img + d_tab, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.use_softmax = use_softmax
        self.out_dim = d_fuse
        self.norm = nn.LayerNorm(d_fuse)
        self.drop = nn.Dropout(dropout)

    def forward(self, img_feat, tab_feat):
        z = torch.cat([img_feat, tab_feat], dim=1)  # [B, d_img+d_tab]
        w = self.gate(z)                            # [B,2]
        w = torch.softmax(w, dim=1) if self.use_softmax else torch.sigmoid(w)

        i = self.proj_img(img_feat)                 # [B,d_fuse]
        t = self.proj_tab(tab_feat)                 # [B,d_fuse]
        fused = w[:, 0:1] * i + w[:, 1:2] * t
        # residual toward average
        fused = self.norm(fused + 0.5*(i + t))
        return self.drop(fused)


class SelfAttentionFusion(nn.Module):
    """Self-attention over the two modality tokens (image, tabular), then mean pooling."""
    def __init__(self, d_img, d_tab, d_out=256, nhead=4, dropout=0.3):
        super().__init__()
        assert d_out % nhead == 0, "d_out must be divisible by nhead"
        self.proj_img = nn.Linear(d_img, d_out)
        self.proj_tab = nn.Linear(d_tab, d_out)
        enc = nn.TransformerEncoderLayer(
            d_model=d_out, nhead=nhead, batch_first=True, dropout=dropout, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=4)
        self.out_dim = d_out
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, img_feat, tab_feat):
        img = self.proj_img(img_feat)
        tab = self.proj_tab(tab_feat)
        tokens = torch.stack([img, tab], dim=1)  # [B,2,D]
        x = self.encoder(tokens)                 # [B,2,D]
        x = x.mean(dim=1)                        # [B,D]
        return self.drop(self.norm(x))


class MultiHeadFusion(nn.Module):
    def __init__(self, d_img, d_tab, num_heads=4, d_model=256, dim_feedforward=512, num_layers=4, dropout=0.3):
        super().__init__()
        # Project both modalities into a shared dimension
        self.img_proj = nn.Linear(d_img, d_model)
        self.tab_proj = nn.Linear(d_tab, d_model)

        # Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # expects [B, seq_len, d_model]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling and output projection
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = d_model

    def forward(self, img_feat, tab_feat):
        """
        img_feat: [B, d_img]
        tab_feat: [B, d_tab]
        """
        B = img_feat.size(0)

        # projection
        img_emb = self.img_proj(img_feat)  # [B, d_model]
        tab_emb = self.tab_proj(tab_feat)  # [B, d_model]

        # Treat the two modalities as a length-2 sequence: [B, 2, d_model]
        x = torch.stack([img_emb, tab_emb], dim=1)

        # Transformer Encoder → [B, seq_len, d_model]
        x = self.transformer(x)

        # Mean pooling over the two modality tokens
        fused = x.mean(dim=1)  # [B, d_model]

        return self.fc(fused)



class FiLMFusion(nn.Module):
    """Modulate image features with tabular features via FiLM (Feature-wise Linear Modulation)."""
    def __init__(self, d_img, d_tab, d_fuse=None, dropout=0.3):
        super().__init__()
        d_fuse = d_fuse or d_img
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(d_tab, d_img*2),
        )
        self.out = nn.Sequential(
            nn.Linear(d_img, d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = d_fuse
        self.norm = nn.LayerNorm(d_img)

    def forward(self, img_feat, tab_feat):
        gamma_beta = self.to_gamma_beta(tab_feat)           # [B, 2*d_img]
        gamma, beta = gamma_beta.chunk(2, dim=1)            # [B,d_img], [B,d_img]
        x = self.norm(img_feat) * (1 + torch.tanh(gamma)) + beta
        return self.out(x)
    

class MultiModalMTLModel(nn.Module):
    def __init__(self, num_tabular_features, backbone_name='resnet50',
                 fusion_type='concat', tabular_type='mlp',
                 pretrained=True, in_chans=3, num_tasks=2,
                 d_fuse=256, dropout=0.3, freeze_backbone=False):
        super().__init__()
        
        
        # Image backbone
        transformer_backbones = [
            'vit_tiny_patch16_224',
            'vit_small_patch16_224',
            'deit_tiny_patch16_224',
            'deit_small_patch16_224',
            'swin_tiny_patch4_window7_224',
            'swin_small_patch4_window7_224',
        ]

        if backbone_name in transformer_backbones:
            self.backbone = create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_chans,
                img_size=512,  # required for transformer backbones
            )
        else:
            self.backbone = create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_chans
            )
            
        if 'vgg' in backbone_name:
            self.img_dim = 4096
        else:
            self.img_dim = getattr(self.backbone, "num_features", None)
            if self.img_dim is None:
                self.img_dim = self.backbone.get_classifier().in_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Tabular encoder
        if tabular_type == 'cnn':
            self.tab_encoder = TabularCNN1D(num_tabular_features, dropout=dropout)
        else:
            self.tab_encoder = TabularMLP(num_tabular_features, dropout=dropout)
        self.tab_dim = self.tab_encoder.out_dim

        # Fusion
        if fusion_type == 'general_attention':
            self.fusion = GeneralAttentionFusion(self.img_dim, self.tab_dim, d_fuse=d_fuse, dropout=dropout)
        elif fusion_type == 'self_attention':
            self.fusion = SelfAttentionFusion(self.img_dim, self.tab_dim, d_out=d_fuse, dropout=dropout)
        elif fusion_type == 'multihead_attention':
            self.fusion = MultiHeadFusion(self.img_dim, self.tab_dim, num_heads=4, dropout=dropout)
        elif fusion_type == 'film':
            self.fusion = FiLMFusion(self.img_dim, self.tab_dim, d_fuse=d_fuse, dropout=dropout)
        else:  # concat
            self.fusion = None

        self.fused_dim = (self.img_dim + self.tab_dim) if self.fusion is None else self.fusion.out_dim
        self.fused_post = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Dropout(dropout),
        )

        # Task-specific heads, so each task keeps its own bias and calibration
        self.head_fvc  = nn.Linear(self.fused_dim, 1)
        self.head_fev1 = nn.Linear(self.fused_dim, 1)

        # Init for new layers
        for m in [self.head_fvc, self.head_fev1]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, imgs, tabs):
        img_feat = self.backbone(imgs)       # [B, img_dim]
        tab_feat = self.tab_encoder(tabs)    # [B, tab_dim]

        if self.fusion is None:
            fused = torch.cat([img_feat, tab_feat], dim=1)
        else:
            fused = self.fusion(img_feat, tab_feat)

        fused = self.fused_post(fused)       # [B, fused_dim]

        logit_fvc  = self.head_fvc(fused)    # [B,1]
        logit_fev1 = self.head_fev1(fused)   # [B,1]
        logits = torch.cat([logit_fvc, logit_fev1], dim=1)  # [B,2]
        return logits
    
    
class SingleModalMTLModel(nn.Module):
    def __init__(self, num_tabular_features=None, backbone_name='resnet50',
                 pretrained=True, in_chans=3, num_tasks=2,
                 d_fuse=256, dropout=0.1, freeze_backbone=False):
        super().__init__()

        # --------------------------
        # Image backbone
        # --------------------------
        transformer_backbones = [
            'vit_tiny_patch16_224',
            'vit_small_patch16_224',
            'deit_tiny_patch16_224',
            'deit_small_patch16_224',
            'swin_tiny_patch4_window7_224',
            'swin_small_patch4_window7_224',
        ]

        if backbone_name in transformer_backbones:
            self.backbone = create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_chans,
                img_size=512,  # required for transformer backbones
                global_pool='avg'
            )
        else:
            self.backbone = create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_chans,
                global_pool='avg'
            )

        # Infer the backbone feature dimension
        if 'vgg' in backbone_name:
            self.img_dim = 4096
        else:
            self.img_dim = getattr(self.backbone, "num_features", None)
            if self.img_dim is None:
                # Fallback for timm models that expose the dimension differently
                try:
                    self.img_dim = self.backbone.get_classifier().in_features
                except Exception:
                    raise RuntimeError(
                        f"Cannot infer feature dim for backbone: {backbone_name}"
                    )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # --------------------------
        # Task-specific heads; image-only, so tabular and fusion are dropped
        # --------------------------
        self.head_fvc  = nn.Linear(self.img_dim, 1)
        self.head_fev1 = nn.Linear(self.img_dim, 1)

        # Head initialization
        for m in [self.head_fvc, self.head_fev1]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, imgs, tabs=None):
        """
        `tabs` is accepted for interface compatibility but ignored (single-modal).
        imgs: [B, C, H, W]
        return: logits [B,2] -> [:,0]=FVC, [:,1]=FEV1
        """
        img_feat = self.backbone(imgs)          # [B, img_dim]
        if img_feat.ndim > 2:  # pool if the backbone returns a feature map
            img_feat = img_feat.mean(dim=[-2, -1])

        x = self.dropout(img_feat)
        logit_fvc  = self.head_fvc(x)           # [B,1]
        logit_fev1 = self.head_fev1(x)          # [B,1]
        logits = torch.cat([logit_fvc, logit_fev1], dim=1)  # [B,2]
        return logits