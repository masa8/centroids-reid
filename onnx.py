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
    

    model = CTLModel.load_from_checkpoint(args.input_file)
    model.forward = model.test_step
    input_sample=torch.randn((args.batch_size,3,256,128))
    dynamic_axes = {'input': {0: 'batch', 2: 'height', 3: 'width'}}
    torch.onnx.export(model, 
            input_sample, 
            args.output_file, 
            export_params=True,
            input_names = ['input'], 
            output_names = ['output'], 
            dynamic_axes = dynamic_axes,
            opset_version=11, 
            verbose=True)

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


