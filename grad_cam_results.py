import cv2
import numpy as np
import torch
from torchvision import models
import sys

def get_last_conv(m):
    convs = filter(lambda k: isinstance(k, torch.nn.Conv2d), m.modules())
    return list(convs)[-1]

class Grad_Cam:
    def __init__(self, model,target_layer_names, use_cuda):
        self.model = model
        self.target = target_layer_names
        self.use_cuda = use_cuda
        self.grad_val = []
        self.feature = [] 
        self.hook = []
        self.img = []
        self.inputs = None
        self._register_hook()
    def get_grad(self,module,input,output):
            self.grad_val.append(output[0].detach())
    def get_feature(self,module,input,output):
            self.feature.append(output.detach())
    def _register_hook(self):
        for i in self.target:
                self.hook.append(i.register_forward_hook(self.get_feature))
                self.hook.append(i.register_full_backward_hook(self.get_grad))

    def _normalize(self,cam):
        h,w,c = self.inputs.shape
        cam = (cam-np.min(cam))/np.max(cam)
        cam = cv2.resize(cam, (w,h))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        cam = heatmap + np.float32(self.inputs)
        cam = cam / np.max(cam)
        return np.uint8(255*cam)

    def remove_hook(self):
        for i in self.hook:
            i.remove()

    def _preprocess_image(self,img):
         means = [0.485, 0.456, 0.406]
         stds = [0.229, 0.224, 0.225]

         preprocessed_img = img.copy()[:, :, ::-1]
         for i in range(3):
             preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
             preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
         preprocessed_img = \
         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
         preprocessed_img = torch.from_numpy(preprocessed_img)
         preprocessed_img.unsqueeze_(0)
         input = preprocessed_img.requires_grad_(True)
         return input

    def __call__(self, img,idx=None):
        
        self.model.zero_grad()
        self.inputs = np.float32(cv2.resize(img, (224, 224))) / 255
        inputs = self._preprocess_image(self.inputs)
        if self.use_cuda:
            inputs = inputs.cuda()
            self.model = self.model.cuda()
        output = self.model(inputs)
        if idx is None:
            idx = np.argmax(output.detach().cpu().numpy()) 
        target = output[0][idx]
        target.backward()
        #computer 
        weights = []
        for i in self.grad_val[::-1]: 
             weights.append(np.mean(i.squeeze().cpu().numpy(),axis=(1,2)))
        for index,j in enumerate(self.feature):
             cam = (j.squeeze().cpu().numpy()*weights[index][:,np.newaxis,np.newaxis]).sum(axis=0)
             cam = np.maximum(cam,0) # relu
             self.img.append(self._normalize(cam))
        return self.img
		
	
if __name__ == '__main__':
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    use_cuda = torch.cuda.is_available()
    # load model
    checkpoint = torch.load(model_path)

    # model = models.resnet50(weights=None, num_classes=4)
    # model = models.vgg19(weights=None, num_classes=4)
    # model = models.googlenet(weights=None, num_classes=4)
    # model = models.regnet_x_16gf(weights=None, num_classes=4)
    # model = models.shufflenet_v2_x2_0(weights=None, num_classes=4)
    # model = models.densenet161(weights=None, num_classes=4)
    # model = models.squeezenet1_0(weights=None, num_classes=4)
    # model = models.alexnet(weights=None, num_classes=4)
    # model = models.resnext101_64x4d(weights=None, num_classes=4)
    # model = models.mnasnet1_0(weights=None, num_classes=4)
    # model = models.efficientnet_b0(weights=None, num_classes=4)
    model = models.vit_b_16(weights=None, num_classes=4)
    # model = models.swin_t(weights=None, num_classes=4)

    model.load_state_dict(checkpoint)
    model.eval()

    #print(model.state_dict)
    img = cv2.imread(img_path, 1)
    m = get_last_conv(model)
    target_layer = [m]
    Grad_cams = Grad_Cam(model,target_layer,use_cuda)
    grad_cam_list  = Grad_cams(img)
    cv2.imwrite("out.jpg",grad_cam_list[0])