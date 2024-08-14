export CUDA_VISIBLE_DEVICES="5"

# exp3-city2map-base 68.2100
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2map-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp3_b2-40k_city2bdd_pretrain-512x1024_base_debug3/best_mIoU_iter_20000.pth 1
# exp3-city2gtav-base  58.6400
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2gtav-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp3_b2-40k_city2bdd_pretrain-512x1024_base_debug3/best_mIoU_iter_20000.pth 1
# exp3-city2syn-base 38.9300
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2syn-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp3_b2-40k_city2bdd_pretrain-512x1024_base_debug3/best_mIoU_iter_20000.pth 1

# exp3-city2all-base 77.7000 | 78.1700 | 68.5500 (no mask)
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2map-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug1/best_mIoU_iter_28000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2gtav-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug1/best_mIoU_iter_28000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2syn-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug1/best_mIoU_iter_28000.pth 1

# exp3-city2all-base 78.0400 | 77.4100 | 64.5700 (mask=0.5, weight=0.1)
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2map-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug9/iter_15000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2map-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug8/iter_13000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2map-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug4/iter_15000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2map-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug4/iter_20000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2map-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug4/iter_25000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2map-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug4/iter_30000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2gtav-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug3/iter_10000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2syn-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug3/iter_10000.pth 1

# exp4
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-20k_gtav2syn-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/gtav_to_bdd/exp4_b2-20k_gtav2bdd-512x1024_tiny_debug1/best_mIoU_iter_10000.pth 1
bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-20k_map2gtav-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/map_to_bdd/exp4_b2-20k_map2bdd-512x1024_tiny_debug4/best_mIoU_iter_20000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-20k_city2gtav-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-20k_city2bdd-512x1024_tiny_debug2/best_mIoU_iter_18000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-20k_city2city-blur-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-20k_city2bdd-512x1024_tiny_debug2/best_mIoU_iter_18000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-20k_city2city-noise-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-20k_city2bdd-512x1024_tiny_debug2/best_mIoU_iter_18000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-20k_city2city-digital-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-20k_city2bdd-512x1024_tiny_debug2/best_mIoU_iter_18000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-20k_city2city-weather-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-20k_city2bdd-512x1024_tiny_debug2/best_mIoU_iter_18000.pth 1
# base
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-20k_city2syn-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-20k_city2bdd-512x1024_base_debug1/best_mIoU_iter_20000.pth 1
# w/ mask
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-40k_city2syn-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-40k_city2bdd-512x1024_tiny_debug3/best_mIoU_iter_28000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-40k_city2map-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-40k_city2bdd-512x1024_tiny_debug3/best_mIoU_iter_28000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-40k_city2city-blur-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-40k_city2bdd-512x1024_tiny_debug3/best_mIoU_iter_28000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-40k_city2city-noise-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-40k_city2bdd-512x1024_tiny_debug3/best_mIoU_iter_28000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-40k_city2city-digital-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-40k_city2bdd-512x1024_tiny_debug3/best_mIoU_iter_28000.pth 1
#bash ./tools/dist_test.sh configs/dg_exp/exp4_b2-40k_city2city-weather-512x1024_tiny.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_bdd/exp4_b2-40k_city2bdd-512x1024_tiny_debug3/best_mIoU_iter_28000.pth 1

# for visualization
#bash ./tools/dist_test.sh configs/dg_exp/exp3_b2-40k_city2syn-512x1024_base.py /data/ljh/segmentation/_LIKE_VMamba/output/city_to_all_pretrain/exp3_b2-40k_city2all_pretrain-512x1024_base_debug3/best_mIoU_iter_33000.pth 1 \
#--show-dir /data/ljh/data/MVM/syn/city2others/base/results/