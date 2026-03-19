import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models_mamba import create_block, RMSNorm, rms_norm_fn, StrideEmbed
from timm.models.layers import trunc_normal_, lecun_normal_
import math
from functools import partial


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class NetMamba(nn.Module):
    def __init__(self, byte_length=1600, stride_size=4, in_chans=1,
                 embed_dim=192, depth=4,
                 decoder_embed_dim=128, decoder_depth=2,
                 num_classes=1000,
                 norm_pix_loss=False,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 bimamba_type="none",
                 is_pretrain=False,
                 device=None, dtype=None,
                 **kwargs):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.is_pretrain = is_pretrain
        self.stride_size = stride_size

        # --------------------------------------------------------------------------
        # NetMamba encoder specifics
        self.patch_embed = StrideEmbed(byte_length, stride_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_cls_token = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, embed_dim))

        # [Nety 创新点 1]：输入端改造 - 1D-CNN (Depthwise Conv)
        # 串联一个轻量级 1D-CNN 提取局部空间关联
        self.cnn_extractor = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=3, padding=1, groups=embed_dim
        )

        # Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=None,
                norm_epsilon=1e-5,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                layer_idx=i,
                if_bimamba=False,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                if_devide_out=True,
                init_layer_scale=None,
            ) for i in range(depth)])
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        # --------------------------------------------------------------------------

        if is_pretrain:
            # --------------------------------------------------------------------------
            # NetMamba decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, decoder_embed_dim))
            decoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
            decoder_inter_dpr = [0.0] + decoder_dpr
            self.decoder_blocks = nn.ModuleList([
                create_block(
                    decoder_embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type=bimamba_type,
                    drop_path=decoder_inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                )
                for i in range(decoder_depth)])
            self.decoder_norm_f = RMSNorm(decoder_embed_dim, eps=1e-5)
            self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)
            # --------------------------------------------------------------------------
        else:
            # --------------------------------------------------------------------------
            # NetMamba classifier specifics

            # [Nety 创新点 2]：输出端融合 (多模态 MLP)
            # 输入维度 19 = PL 序列(10) + IAT 序列(9)
            self.stat_mlp = nn.Sequential(
                nn.Linear(19, 64),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Linear(64, 128),
                nn.LayerNorm(128)
            )
            # 最终分类头输入维度 = Mamba特征 (embed_dim) + 统计特征 (128)
            fused_dim = self.num_features + 128
            self.head = nn.Linear(fused_dim, num_classes) if num_classes > 0 else nn.Identity()
            # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.patch_embed.apply(segm_init_weights)
        if not self.is_pretrain:
            self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.is_pretrain:
            trunc_normal_(self.decoder_pos_embed, std=.02)
            trunc_normal_(self.mask_token, std=.02)

        self.apply(partial(_init_weights, n_layer=depth, ))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def stride_patchify(self, imgs):
        B, C, H, W = imgs.shape
        assert C == 1, "Input images should be grayscale"
        stride_size = self.stride_size
        x = imgs.reshape(B, H * W // stride_size, stride_size)
        return x

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, if_mask=True, ):
        # embed patches
        B, C, H, W = x.shape
        x = self.patch_embed(x.reshape(B, C, -1))

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, :-1, :]

        # [Nety 创新层] 1D-CNN 特征提取
        # 注意：Conv1d 接收的维度是 [Batch, Channel, Length]
        x = x.transpose(1, 2)
        x = self.cnn_extractor(x)
        x = x.transpose(1, 2)

        # masking
        if if_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, -1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.pos_drop(x)

        # apply Mamba blocks
        residual = None
        hidden_states = x
        for blk in self.blocks:
            hidden_states, residual = blk(hidden_states, residual)
        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
        if if_mask:
            return x, mask, ids_restore
        else:
            return x

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        visible_tokens = x[:, :-1, :]
        x_ = torch.cat([visible_tokens, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x_, x[:, -1:, :]], dim=1)
        x = x + self.decoder_pos_embed

        residual = None
        hidden_states = x
        for blk in self.decoder_blocks:
            hidden_states, residual = blk(hidden_states, residual)
        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.decoder_norm_f.weight,
            self.decoder_norm_f.bias,
            eps=self.decoder_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )

        x = self.decoder_pred(x)
        x = x[:, :-1, :]
        return x

    def forward_rec_loss(self, imgs, pred, mask):
        target = self.stride_patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    # [Nety 核心接口改造]：接收 pl 和 iat 特征
    def forward(self, imgs, mask_ratio=0.9, pl=None, iat=None):
        B, C, H, W = imgs.shape
        assert C == 1, "Input images should be grayscale"

        if self.is_pretrain:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio=mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_rec_loss(imgs, pred, mask)
            return loss, pred, mask
        else:
            x = self.forward_encoder(imgs, mask_ratio=mask_ratio, if_mask=False)
            v_sem = x[:, -1, :]  # 提取 Mamba 语义特征

            # 多模态拼接融合逻辑
            if pl is not None and iat is not None:
                stat_concat = torch.cat([pl, iat], dim=1)  # [B, 19]
                v_stat = self.stat_mlp(stat_concat)  # [B, 128]
                v_fused = torch.cat([v_sem, v_stat], dim=1)  # Concat 操作
                return self.head(v_fused)
            else:
                # 兼容不带统计特征的输入（降级模式，防止预训练等环节报错）
                dummy_stat = torch.zeros(B, 128, device=imgs.device)
                v_fused = torch.cat([v_sem, dummy_stat], dim=1)
                return self.head(v_fused)


def net_mamba_pretrain(**kwargs):
    model = NetMamba(
        is_pretrain=True, stride_size=4, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2, **kwargs)
    return model


def net_mamba_classifier(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=256, depth=4,
        **kwargs)
    return model


def net_mamba_bl400_pretrain(**kwargs):
    model = NetMamba(
        is_pretrain=True, stride_size=4, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2,
        byte_length=400, **kwargs)
    return model


def net_mamba_bl400_classifier(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=256, depth=4,
        byte_length=400, **kwargs)
    return model


def net_mamba_bl800_pretrain(**kwargs):
    model = NetMamba(
        is_pretrain=True, stride_size=4, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2,
        byte_length=800, **kwargs)
    return model


def net_mamba_bl800_classifier(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=256, depth=4,
        byte_length=800, **kwargs)
    return model