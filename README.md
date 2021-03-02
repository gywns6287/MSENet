# MSENet: Bridge connected Network for beef marbling score estimation and eye-muscle area segmentation

 
## Object

We proposed the MSENet for end-to-end networks without any pre
or post processing step, implementing segmentation of beef eye-muscle area and estimation of marblingscore.In particular, the bridge block in the MSENet interlinks a scoring module and segmentation module, trans-forming the segmentation information to scoring module for way to improve the estimation performance. We applied it to connect segmentation module with ResNet50 and Xception for automated beef MS grading. Bridge block is also simple to use for any other automated systems demanding both classification and segmentation task since it doesn’t need to create a new deep learning network but simply combines two existing networks.

##  Installation
The pre-trained weight must exist as `resnet_weights.h5` and  `xception_weights.h5` in the path where `main.py` is located. pre-trained weight can be downloaded at https://drive.google.com/drive/folders/1E8x43bGvKinJRwiTPb-A6eOr9pMvvxDi.
## Model summary

![fig4](https://user-images.githubusercontent.com/71325306/106382944-3b282480-6406-11eb-81c7-1393a09fe952.png)
  
our models were implemented by **tensorflow 2.3** and **keras**

 #### Model summary

1. This model is end-to-end networks for eye-muscle area segmentation and marbling score estimation.

2. MSENet-ResNet50 designed by ResNet50 (He et al., 2016) for the scoring module, while MSENet-Xception use Xception (Chollet, 2017).

3. Model code can be found in `model.py`.

####   Execution for MSENet-ResNet50
```
python main.py --img example --out example_results --model resnet
```

####   Execution for MSENet-Xception
```
python main.py --img example --out example_results --model xception
```
  

## Model performance

Model was trained by 8,201 beef images and ground truth mask and true MS,
and test set contain 1,024 images.

  

|model|+/-2 Acc|mIOU|
|-----|--------|----|
|Bridge-ResNet50|99.7%|0.973 |
|Bridge-Xception|99.8%|0.973 |

![fig9](https://user-images.githubusercontent.com/71325306/105946387-865ed200-60aa-11eb-8452-eb0b51f2db5f.png)
