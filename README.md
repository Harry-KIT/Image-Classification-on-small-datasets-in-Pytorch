# Image-Classification-on-small-datasets-in-Pytorch
This is a package with state of the art methods for Explainable AI for computer vision.
This can be used for observing processes while developing models and diagnosing model predictions.
The aim is also to serve as a benchmark of algorithms and metrics for research of new explainability methods.

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

| rhino | buffalo | elephant | zebra |
| ---------------------------------------------------------------|--------------------|-----------------------------------------------------------------------------|
 <img src="https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/dog.jpg?raw=true" width="200" height="112"> | <img src="https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cat.jpg?raw=true" width="200" height="112"> | <img src="https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cam_gb_dog.jpg?raw=true" width="200" height="112"> | <img src="https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cam_gb_dog.jpg?raw=true" width="200" height="112"> |

## Quantitative results
#### Squeezenet:
| Training Accuracy | Training Loss | Confusion Matrix |
| -----------------|-----------------------|-----------------|
| <img src="./examples/both_detection.png" width="256" height="256"> | <img src="./examples/cars_segmentation.png" width="256" height="256"> |<img src="./examples/cars_segmentation.png" width="256" height="256"> |

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

| Models  |            Grad-CAM results                 |
| ---------|-------|----------|------------|------------|
| VGG             | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.
jpg) |
| AlexNet         | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |                                                                                        
| ResNet          | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |                                                                                          
| Squeezenet      | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |                                                                                       
| Densenet        | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |                                                                                           
| Shufflenet      | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |                                                                                          
| RegNet          | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |                                                                                           
| ResNeXt         | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |                                                                                           
| EfficientNet_v2 | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |                                                                                           



----------
## ResNet-50 Grad-CAM results by every layers
| Model     | laye-1 | laye-2 | layer-3 | layer-4 |
| --------- |--------|--------|---------|---------|
| ResNet-50 | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) | ![](./examples/resnet50_dog_gradcam_cam.jpg) |![](./examples/resnet50_dog_gradcam_cam.jpg) |

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

## P.S
Please give ⭐ to this repo if you find it useful. Stay tuned!