export CUDA_VISIBLE_DEVICES="2"


#/data/ljh/data/datasets/synthia/RGB/val_1_80/ \
#/data/ljh/data/datasets/synthia/RGB/val/0000726.png \
# city-to-syn visualization
python inner_feat.py \
configs/dg_exp/exp4_b2-20k_city2bdd-512x1024_tiny.py \
/data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-20k_city2bdd-512x1024_tiny_debug2/best_mIoU_iter_18000.pth \
--work-dir /data/ljh/data/MaskViM/city/