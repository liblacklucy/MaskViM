export CUDA_VISIBLE_DEVICES="0"
export LD_LIBRARY_PATH=/home/ljh/miniconda3/envs/dgss/lib/:$LD_LIBRARY_PATH

python feat_vis.py \
configs/dg_exp/exp4_b2-20k_city2bdd-512x1024_tiny.py \
/data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-20k_city2bdd-512x1024_tiny_debug2/best_mIoU_iter_18000.pth \
--work-dir /data/ljh/data/MaskViM/bdd/feat_vis/exp4/ --dataset-name bdd