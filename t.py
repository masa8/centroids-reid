import argparse
from config import cfg
from train_ctl_model import CTLModel
import torch
import pytorch_lightning as pl
import onnx
from inference.inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    make_inference_data_loader,
    run_inference,
)
from config import cfg
from datasets.market1501 import Market1501


def main(args):
    
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 1
    print(cfg.DATALOADER.NUM_WORKERS)
    val_loader = make_inference_data_loader(cfg, './data/market1501/bounding_box_train/', ImageDataset)
    model = CTLModel.load_from_checkpoint("./models/market1501_resnet50_256_128_epoch_120.ckpt")
    print("=======")
    model.eval()
    with torch.no_grad():
        for pos, batch in enumerate(val_loader):
            # x = list[images, _, paths]
            x, _, path = batch
            y = model.backbone(x)
            print(y)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model to ONNX")
    parser.add_argument("--output_file", default="centroid-reid_resnet50_256_128_epoch_120.onnx", help="name of onnx", type=str)
    parser.add_argument("--input_file", default="./models/market1501_resnet50_256_128_epoch_120.ckpt", type=str)
    parser.add_argument("--batch_size", default=1, help="batch_size", type=int)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )    
   
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(e.args)



