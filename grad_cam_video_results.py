import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import models
import sys
import glob
from tqdm import tqdm

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
                self.hook.append(i.register_backward_hook(self.get_grad))

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

def extract_frames_from_video(input, output):
    vidcap = cv2.VideoCapture(input)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
        

    print(f"Video: {input}")
    print(f"Extracting frames to {output}")

    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(
            os.path.join(output, f"frame-{count:04}.jpg"),
            image,
        )
        success, image = vidcap.read()
        count += 1
    return output, fps
    
	
if __name__ == '__main__':
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    video_frames_folder = 'video_frames'
    heatmap_frames = 'heatmap_output_frames'
    video_folder = 'video_folder'

    os.makedirs(video_frames_folder, exist_ok=True)
    os.makedirs(heatmap_frames, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    # load model
    checkpoint = torch.load(model_path)

    # model = models.vgg19(weights=None, num_classes=4)
    # model = models.densenet161(weights=None, num_classes=4)
    # model = models.resnet50(weights=None, num_classes=4)
    model = models.squeezenet1_0(weights=None, num_classes=4)
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

    model.load_state_dict(checkpoint)
    model.eval()

    #print(model.state_dict)
    video_output_frames, fps = extract_frames_from_video(video_path, video_frames_folder)

    video_output_frames = sorted(glob.glob(os.path.join(video_output_frames, "*.jpg")))
    genral_caunt = len(video_output_frames)
    count = 0
    while genral_caunt > 0:
        img = cv2.imread(video_output_frames[count], 1)
        m = get_last_conv(model)
        target_layer = [m]
        Grad_cams = Grad_Cam(model,target_layer,use_cuda)
        grad_cam_list  = Grad_cams(img)
        cv2.imwrite(
            os.path.join(heatmap_frames, f"att-{count:04}.jpg"),
            grad_cam_list[0]
        )
        genral_caunt -=1
        count +=1
    
    
    img_array = []
    heatmap_frames = sorted(glob.glob(os.path.join(heatmap_frames, "*.jpg")))
    with open(heatmap_frames[0], "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        size = (img.width, img.height)
        img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    print(f"Generating video {size} to {video_folder}")

    for filename in tqdm(heatmap_frames[1:]):
        with open(filename, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    out = cv2.VideoWriter(
        os.path.join(video_folder, "video." + "mp4"),
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps,
        size,
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Done")
