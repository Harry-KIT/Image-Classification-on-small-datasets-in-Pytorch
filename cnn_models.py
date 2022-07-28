from torchvision import models
from torch import nn
from config import *

def initialize_model(model_name, num_classes, feature_extract=True): # weights=None

    model = None
    
    if model_name == "densenet":
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "vgg":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
    
    elif model_name == "resnet":
        
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    
    elif model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
    
    elif model_name == "resnext":
        
        model = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "mnasnet":
        model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "regnet":
        model = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "ViT":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    elif model_name == "Swin":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        print(model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
    
    else:
        print("Invalid model name")
        exit()
    
    # Gather the parameters to be optimized/updated in this run.
    params_to_update = set_params_to_update(model, feature_extract)
    
    
    return model, params_to_update


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def set_params_to_update(model, feature_extract):
    params_to_update = []
    if feature_extract:
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model.parameters()
    return params_to_update