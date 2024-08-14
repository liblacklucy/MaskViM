_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/cityscapes_to_bdd_pretrain.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
# find_unused_parameters = True  # for debug, set TORCH_DISTRIBUTED_DEBUG=DETAIL before bash
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='Exp2',
        pretrained="/data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd_pretrain/exp2_4xb4-20k_pretrain-512x1024_tiny_debug1/best_mIoU_iter_20000.pth",
        extra=dict(
            depths=(int(4*1), int(4*1), int(4*1)),
            stage1=dict(
                num_modules=1,
                num_branches=1,
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=1,  # 4,
                num_branches=3,
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=1,  # 3,
                num_branches=4,
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144))
        ),
        drop_path_rate=0.3,
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        # forward_type="v0", # if you want exactly the same
    ),)

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
            # 'norm': dict(decay_mult=0.)
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
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader