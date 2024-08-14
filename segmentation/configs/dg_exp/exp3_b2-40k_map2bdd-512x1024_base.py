_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/mapillary_to_bdd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
# find_unused_parameters = True  # for debug, set TORCH_DISTRIBUTED_DEBUG=DETAIL before bash
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    type='SegExp3',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="/data/ljh/pretrained/vssmbase_dp05_ckpt_epoch_260.pth",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=128,
        depths=(2, 2, 27, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        drop_path_rate=0.3,
        patch_norm=True
        # forward_type="v0", # if you want exactly the same
    ),
    # decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=19),
    decode_head=dict(
        _delete_=True,
        type='DecoderExp3',
        # token mask settings =========
        gamma=4,
        input_resolution=crop_size,  # same as crop
        patch_size=4,  # same as encoder
        t_mask=True,
        in_channels=[128, 256, 512, 1024],
        num_classes=19,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        # vss block settings =======
        depths=[1, 1, 1],
        channels=256,
        dropout_ratio=0.1,
        align_corners=False,
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='MaskTokenLoss', keep_ratio=0.9, loss_weight=1.0),
        ]
        # init_cfg=None,  # for delattr(self, 'conv_seg')
    ),
    auxiliary_head=dict(in_channels=512, in_index=2, num_classes=19,)
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader