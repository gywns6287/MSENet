
# MSENet: End-to-end network for the beef marbling score estimation of Korean native cattle

 
## Object

We propose the MSENet for end-to-end networks, which simultaneously performs marbling score estimation and eye muscle area segmentation. The proposed MSENet include a segmentation module, a bridge block, and a marbling scoring module. In particular, the segmentation module yields multi-scale attention maps for eye muscle area and the bridge block is transfers the multi-scale attentions maps to the scoring module. Unlike other previous approaches, MSENet is trained on a new large-scale beef image dataset (more than 10000), called a Hanwoo dataset.

##  Installation
The pre-trained weight must exist as `resnet_weights.h5` and  `xception_weights.h5` in the path where `main.py` is located. pre-trained weight can be downloaded at https://drive.google.com/drive/folders/1E8x43bGvKinJRwiTPb-A6eOr9pMvvxDi.
## Model summary

![fig4](https://user-images.githubusercontent.com/71325306/129844067-98c71703-c483-4c7f-ab37-b95a3902944c.png)
  
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
and test set contain 1,024 images. For MSENet-ResNet50, we design the convolution blocks in the scoring module by employing ResNet50 (He et al., 2016), while the structure of Xception (Chollet, 2017) is adopted for the convolution blocks in MSENet-Xception

  

|model|Corr|mIOU|
|-----|--------|----|
|MSENet-ResNet50|0.95|0.973 |
|MSENet-Xception|0.952|0.973 |

![joint_raw](https://user-images.githubusercontent.com/71325306/119126744-5e86e600-ba6e-11eb-8b07-1e6c155fa7e6.png)
![joint_results](https://user-images.githubusercontent.com/71325306/119126717-5464e780-ba6e-11eb-9a01-1dbb60e10e51.png)

## Reference

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition, (pp. 770–778)
- Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. Proceedings of the IEEE conference on computer vision and pattern recognition, (pp. 1251–1258)

