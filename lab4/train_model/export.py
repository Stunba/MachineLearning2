import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from model import Model
from onnx_coreml import convert

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint, e.g. ./logs/model-100.pth')
parser.add_argument('input', type=str, help='path to input image')

def export(path_to_checkpoint_file, path_to_input_image):
    # Step 0 - (b) Create model or Load from dist
    model = Model()
    model.restore(path_to_checkpoint_file)

    with torch.no_grad():

        transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = Image.open(path_to_input_image)
        image = image.convert('RGB')
        image = transform(image)
        images = image.unsqueeze(dim=0)

        outputs = model.eval()(images)

        # Step 1 - PyTorch to ONNX model
        torch.onnx.export(model, images, './model.onnx', example_outputs=outputs)

        # Step 2 - ONNX to CoreML model
        mlmodel = convert(model='./model.onnx', minimum_ios_deployment_target='13')
        # Save converted CoreML model
        mlmodel.save('SVNHModel.mlmodel')

def main(args):
    path_to_checkpoint_file = args.checkpoint
    path_to_input_image = args.input

    export(path_to_checkpoint_file, path_to_input_image)


if __name__ == '__main__':
    main(parser.parse_args())