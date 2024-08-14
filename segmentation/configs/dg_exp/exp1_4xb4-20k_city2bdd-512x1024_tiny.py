_base_ = [
    './swin-tiny-patch4-window7-in1k-pre_upernet_4xb4-20k_city2bdd-512x1024.py'
]
model = dict(
    backbone=dict(
        type='Exp1',
        out_indices=(0, 1, 2, 3),
        # pretrained="/data/ljh/pretrained/vssmtiny_dp01_ckpt_epoch_292.pth",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=96,
        depths=(2, 2, 9, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        # forward_type="v0", # if you want exactly the same
    ),)


