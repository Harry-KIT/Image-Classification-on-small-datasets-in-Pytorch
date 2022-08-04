# Image-Classification-on-small-datasets-in-Pytorch
In this repo, we will learn how to finetune and feature extract the torchvision models, all of which have been pretrained on the 1000-class Imagenet dataset. This repo will give an indepth look at how to work with modern CNN architectures, and will build an intuition for finetuning any PyTorch model. Since each model architecture is different, there is no boilerplate finetuning code that will work in all scenarios. Rather, the researcher must look at the existing architecture and make custom adjustments for each model.

## CNN and Vision Transformer models
| Models             | Paper Links                                                                                                                 |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------|
| VGG                | https://arxiv.org/abs/1409.1556                                                                                             |
| AlexNet            | https://arxiv.org/abs/1803.01164                                                                                            |
| ResNet             | https://arxiv.org/abs/1512.03385                                                                                            |
| Squeezenet         | https://arxiv.org/abs/1602.07360v4                                                                                          |
| Densenet           | https://arxiv.org/abs/1608.06993                                                                                            |
| Googlenet          | https://arxiv.org/abs/1409.4842                                                                                             |
| Shufflenet         | https://arxiv.org/abs/1707.01083                                                                                            |
| MNASNet            | https://arxiv.org/abs/1807.11626                                                                                            |
| RegNet             | https://arxiv.org/abs/2003.13678                                                                                            |
| ResNeXt            | https://arxiv.org/abs/1611.05431                                                                                            |
| EfficientNet       | https://arxiv.org/abs/1905.11946                                                                                            |
| EfficientNet_v2    | https://arxiv.org/abs/2104.00298                                                                                            |
| ViT                | https://arxiv.org/abs/2010.11929                                                                                            |
| Swin               | https://arxiv.org/abs/2103.14030                                                                                            |

## Class names

| rhino | buffalo |
| ------|---------|
 <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/samples1/rhino_sample_image.jpg?raw=true" width="375" height="211"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/samples1/buffalo_sample_image.jpg?raw=true" width="375" height="211"> |
 
| elephant | zebra |
| ---------|-------|
 <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/samples1/elephant_sample_image.jpg?raw=true" width="375" height="211"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/samples1/zebra_sample_image.jpg?raw=true" width="375" height="211"> | 


## Quantitative results
#### Squeezenet:
| Training Accuracy | Training Loss | Confusion Matrix |
| -----------------|-----------------------|-----------------|
| <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/squeezenet-gradresults/acc.png" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/squeezenet-gradresults/loss.png" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/squeezenet-gradresults/confusion_matrix.png" width="256" height="256"> |

------------

| class names | precision | recall | f1-score | support |
| ------------|-----------|--------|----------|---------|
| buffalo     |   0.93    |  0.97  |  0.95    |   113   |
| elephant    |   0.91    |  0.93  |  0.92    |   113   |
| rhino       |   0.94    |  0.91  |  0.93    |   113   |
| zebra       |   0.98    |  0.96  |  0.97    |   113   |
||
| accuracy    |           |        |  0.94    |   452   |
| macro avg   |   0.94    |  0.94  |  0.94    |   452   |
| weighted avg|   0.94    |  0.94  |  0.94    |   452   |

## Qualitative Results 
#### Grad-CAM images

| Models   |  rhino   | buffalo |  elephant   |   zebra      |
| ---------|-------|----------|------------|------------|
| VGG             | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/vgg-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/vgg-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/vgg-gradresults/elephant_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/vgg-gradresults/zebra_out.jpg" width="256" height="256"> |
| AlexNet         | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/alexnet-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/alexnet-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/alexnet-gradresults/elephant_out.jpg" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/alexnet-gradresults/zebra_out.jpg" width="256" height="256"> |                                                                                        
| ResNet          | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/resnet-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/resnet-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/resnet-gradresults/elephant_out.jpg" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/resnet-gradresults/zebra_out.jpg" width="256" height="256"> |                                                                                          
| Squeezenet      | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/squeezenet-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/squeezenet-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/squeezenet-gradresults/elephant_out.jpg" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/squeezenet-gradresults/zebra_out.jpg" width="256" height="256"> |                                                                                       
| Densenet        | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/densenet-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/densenet-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/densenet-gradresults/elephant_out.jpg" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/densenet-gradresults/zebra_out.jpg" width="256" height="256"> |                                                                                           
| Shufflenet      | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/shufflenet-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/shufflenet-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/shufflenet-gradresults/elephant_out.jpg" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/shufflenet-gradresults/zebra_out.jpg" width="256" height="256"> |                                                                                          
| RegNet          | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/regnet-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/regnet-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/regnet-gradresults/elephant_out.jpg" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/regnet-gradresults/zebra_out.jpg" width="256" height="256"> |                                                                                           
| ResNeXt         | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/resnext-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/resnext-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/resnext-gradresults/elephant_out.jpg" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/resnext-gradresults/zebra_out.jpg" width="256" height="256"> |                                                                                           
|EfficientNet_v2| <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/efficientnetv2-gradresults/rhino_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/efficientnetv2-gradresults/buffalo_out.jpg" width="256" height="256"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/efficientnetv2-gradresults/elephant_out.jpg" width="256" height="256"> |<img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/examples/efficientnetv2-gradresults/zebra_out.jpg" width="256" height="256"> |                                                                                      
----------

# Using codes

### split dataset
`python split_dataset.py`

```python

  ├── Dataset folder 
    ├── train
        ├── class1
              ├── 1111.png
              ├── 2222.png
        ├── class2
              ├── 1111.png
              ├── 2222.png
    ├── val
        ├── class1
              ├── 1111.png
              ├── 2222.png
        ├── class2
              ├── 1111.png
              ├── 2222.png
```

### Train model
`python train.py`

```python
# uncomment one of the CNN models in train.py and run 

def train():
    data_loader = Dataloader(data_dir='dataset', image_size=IMAGE_SIZE, batch_size=32)

    '''---Choose model---'''

    model, params_to_update = cnn_models.initialize_model(model_name='vgg', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='densenet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='resnet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='squeezenet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='alexnet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='resnext', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='googlenet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='mnasnet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='regnet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='shufflenet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='efficientnet_b7', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='efficientnet_v2', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='ViT', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='Swin', num_classes=4, feature_extract=True)
```

### Generate GRAD-CAM images
`python grad_cam_results.py results/wild-animal-dataset.pth samples1/zebra_sample_image.jpg`

```python
# uncomment one of the CNN models in grad_cam_results.py and run 

model = models.vgg19(weights=None, num_classes=4)
# model = models.densenet161(weights=None, num_classes=4)
# model = models.resnet50(weights=None, num_classes=4)
# model = models.squeezenet1_0(weights=None, num_classes=4)
# model = models.alexnet(weights=None, num_classes=4)
# model = models.resnext101_64x4d(weights=None, num_classes=4)
# model = models.googlenet(weights=None, num_classes=4)
# model = models.mnasnet1_0(weights=None, num_classes=4)
# model = models.regnet_x_16gf(weights=None, num_classes=4)
# model = models.shufflenet_v2_x2_0(weights=None, num_classes=4)
# model = models.efficientnet_b7(weights=None, num_classes=4)
# model = models.efficientnet_v2_s(weights=None, num_classes=4)
# model = models.vit_l_16(weights=None, num_classes=4)
# model = models.swin_b(weights=None, num_classes=4)
```

----------
| Original | Grad-CAM-Video |
| ------|---------|
 <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/samples1/Zebra-gif-resized.gif?raw=true" width="375" height="211"> | <img src="https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch/blob/main/samples1/Zebra-attn-gif-resized.gif?raw=true" width="375" height="211"> |
----------

## P.S
Please give ⭐ to this repo if you find it useful. Stay tuned!
