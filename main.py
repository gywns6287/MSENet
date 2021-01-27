
#Config
input_shape = (224,224,3)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img")
parser.add_argument("--out",default='.')
parser.add_argument('--model')
parser.add_argument("--device", default = '0')
args = parser.parse_args()
#python main.py --img example --out example_results --model xception

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.device

from model import *
from utils import *

#model construct
if args.model == 'resnet':
    model = Bridge_Resnet(input_shape = input_shape, bridge = True)
    model.load_weights('resnet_weights.h5')

if args.model == 'xception':
    model = Bridge_Xception(input_shape = input_shape, bridge = True)
    model.load_weights('xception_weights.h5')

#Estimation
prediction(model, args.img, args.out,input_shape = input_shape[:-1])

