import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models_mamba import create_block, RMSNorm, rms_norm_fn, StrideEmbed
from timm.models.layers import trunc_normal_, lecun_normal_
import math
from functools import partial

def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        lecun_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

class NetMamba(nn.Module):
    def __init__(self, byte_length=1600, stride_size=4, in_chans=1, embed_dim=192, depth=4, num_classes=20, drop_path_rate=0.1, bimamba_type="none", is_pretrain=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.is_pretrain = is_pretrain
        self.stride_size = stride_size
        self.patch_embed = StrideEmbed(byte_length, stride_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # [Nety 保留点]：明确保留 1D-CNN 空间增强模块
        self.cnn_extractor = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=3, padding=1, groups=embed_dim
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=0.)
        self.blocks = nn.ModuleList([
            create_block(embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i, drop_path=[x.item() for x in torch.linspace(0, drop_path_rate, depth)][i])
            for i in range(depth)])
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)

        # [Nety 消融点]：移除 stat_mlp，分类头只接收语义特征维度 (embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(partial(_init_weights, n_layer=depth))

    def forward_encoder(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x.reshape(B, C, -1))
        x = x + self.pos_embed[:, :-1, :]

        # [Nety 保留点] 1D-CNN 提取局部空间关联
        x = x.transpose(1, 2)
        x = self.cnn_extractor(x)
        x = x.transpose(1, 2)

        cls_token = self.cls_token + self.pos_embed[:, -1, :]
        x = torch.cat((x, cls_token.expand(x.shape[0], -1, -1)), dim=1)
        x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for blk in self.blocks:
            hidden_states, residual = blk(hidden_states, residual)
        return rms_norm_fn(self.drop_path(hidden_states), self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps, residual=residual, prenorm=False, residual_in_fp32=True)

    def forward(self, imgs, **kwargs):
        # [Nety 消融点]：仅进行图像张量的空间语义提取，无多模态融合
        x = self.forward_encoder(imgs)
        v_sem = x[:, -1, :]
        return self.head(v_sem)

def net_mamba_classifier(**kwargs):
    return NetMamba(is_pretrain=False, stride_size=4, embed_dim=256, depth=4, **kwargs)
