# training
#python3 train_ctl_model.py --config_file="configs/256_resnet50.yml" DATASETS.NAMES 'market1501' DATASETS.ROOT_DIR './data/' SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 128 SOLVER.BASE_LR 0.00035 OUTPUT_DIR './logs/' SOLVER.EVAL_PERIOD 40 TEST.ONLY_TEST True MODEL.PRETRAIN_PATH "./models/"

# embeddings
python3 inference/create_embeddings.py --config_file="configs/256_resnet50.yml" DATASETS.ROOT_DIR './data/market1501/bounding_box_train/' TEST.IMS_PER_BATCH 1 OUTPUT_DIR '.' TEST.ONLY_TEST True MODEL.PRETRAIN_PATH './models/market1501_resnet50_256_128_epoch_120.ckpt'
