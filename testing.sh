# training
#python3 train_ctl_model.py --config_file="configs/256_resnet50.yml" DATASETS.NAMES 'market1501' DATASETS.ROOT_DIR './data/' SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 128 SOLVER.BASE_LR 0.00035 OUTPUT_DIR './logs/' SOLVER.EVAL_PERIOD 40 TEST.ONLY_TEST True MODEL.PRETRAIN_PATH "./models/"

# testing
python3 train_ctl_model.py  TEST.ONLY_TEST True MODEL.PRETRAIN_PATH "./models/resnet50-19c8e357.pth"
