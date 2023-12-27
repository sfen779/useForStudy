# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import argparse
import io
import sys
import onnx
import onnxruntime

sys.path.append('.')

from models.PosterV2_7cls import *
from main import *


onnx_model = onnx.load("posterV2_7cls.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("posterV2_7cls.onnx",providers = ["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_path = "test.jpg"

from PIL import Image

img_pil = Image.open(img_path)
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225]),
                                                                     transforms.RandomErasing(p=1, scale=(0.05, 0.05))])
input_img = test_transform(img_pil)
input_tensor = input_img.unsqueeze(0).numpy()
input_tensor.shape

ort_inputs = {'input':input_tensor}
ort_outputs = ort_session.run(['output'],ort_inputs)
print(ort_outputs)

