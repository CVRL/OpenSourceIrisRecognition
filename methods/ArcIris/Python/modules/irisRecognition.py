import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
import math
from math import pi
from torchvision import models
from torchvision.models.convnext import LayerNorm2d
from functools import partial
from modules.network import conv, fclayer, iresnet100

class irisRecognition(object):
    def __init__(self, cfg):
        # cParsing the config file
        self.cuda = cfg["cuda"]
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.polar_height = cfg["polar_height"]
        self.polar_width = cfg["polar_width"]
        
        self.circle_model_path = cfg["circle_model_path"]
        self.nn_model_path = cfg["nn_model_path"]

        self.NET_INPUT_SIZE = (320,240)

        with torch.inference_mode():
            self.circle_model = models.resnet18()
            self.circle_model.avgpool = conv(in_channels=512, out_n=6)
            self.circle_model.fc = fclayer(out_n=6)
            try:
                self.circle_model.load_state_dict(torch.load(self.circle_model_path, map_location=self.device))
            except AssertionError:
                    print("assertion error")
                    self.circle_model.load_state_dict(torch.load(self.circle_model_path,
                        map_location = lambda storage, loc: storage))
            self.circle_model = self.circle_model.float().to(self.device)
            self.circle_model = self.circle_model.eval()
            
            self.nn_model = iresnet100(pretrained=False, progress=False)
            state_dict = torch.load(self.nn_model_path, map_location=self.device)
            new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            self.nn_model.load_state_dict(new_state_dict, strict=True)
            self.nn_model = self.nn_model.float().to(self.device)
            self.nn_model = self.nn_model.eval()

            self.input_transform = Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])
                    
        self.ISO_RES = (640,480)

    # converts non-ISO images into ISO dimensions
    def fix_image(self, image):
        w, h = image.size
        aspect_ratio = float(w)/float(h)
        if aspect_ratio >= 1.333 and aspect_ratio <= 1.334:
            result_im = image.copy().resize(self.ISO_RES)
        elif aspect_ratio < 1.333:
            w_new = h * (4.0/3.0)
            w_pad = (w_new - w) / 2
            result_im = Image.new(image.mode, (int(w_new), h), 127)
            result_im.paste(image, (int(w_pad), 0))
            result_im = result_im.resize(self.ISO_RES)
        else:
            h_new = w * (3.0/4.0)
            h_pad = (h_new - h) / 2
            result_im = Image.new(image.mode, (w, int(h_new)), 127)
            result_im.paste(image, (0, int(h_pad)))
            result_im = result_im.resize(self.ISO_RES)
        return result_im
    
    @torch.inference_mode()
    def circApprox(self, image):

        w,h = image.size

        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        inp_xyr_t = self.circle_model(Variable(self.input_transform(image).unsqueeze(0).repeat(1,3,1,1).to(self.device)))

        #Circle params
        diag = math.sqrt(w**2 + h**2)
        inp_xyr = inp_xyr_t.tolist()[0]
        pupil_x = (inp_xyr[0] * w)
        pupil_y = (inp_xyr[1] * h)
        pupil_r = (inp_xyr[2] * 0.5 * 0.8 * diag)
        iris_x = (inp_xyr[3] * w)
        iris_y = (inp_xyr[4] * h)
        iris_r = (inp_xyr[5] * 0.5 * diag)

        return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])
    
    # New Rubbersheet model-based Cartesian-to-polar transformation. It has support for bilinear interpolation
    @torch.inference_mode()
    def grid_sample(self, input, grid, interp_mode):  #helper function for new interpolation
        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, align_corners=True, padding_mode='border')

    @torch.inference_mode()
    def cartToPol_torch(self, image, pupil_xyr, iris_xyr, interpolation='bilinear'): # 
        if pupil_xyr is None or iris_xyr is None:
            return None, None
        
        image = torch.tensor(np.array(image)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        width = image.shape[3]
        height = image.shape[2]

        polar_height = self.polar_height
        polar_width = self.polar_width
        pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float().to(self.device)
        iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float().to(self.device)
        
        theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).to(self.device)
        pxCirclePoints = (pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device) #b x 512
        pyCirclePoints = (pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device)  #b x 512
        
        ixCirclePoints = (iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device)  #b x 512
        iyCirclePoints = (iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device) #b x 512

        radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).to(self.device)  #64 x 1
        
        pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        
        ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

        x = (pxCoords + ixCoords).float()
        x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512

        y = (pyCoords + iyCoords).float()
        y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512

        grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).to(self.device)

        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode=interpolation)
        image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)

        return (image_polar[0][0].cpu().numpy()).astype(np.uint8)

    # (Fixed) Old implementation of Rubbersheet model-based Cartesian-to-polar transformation that uses nearest neighbor interpolation
    def cartToPol(self, image, pupil_xyr, iris_xyr):
        
        if pupil_xyr is None:
            return None, None
       
        image = np.array(image)
        height, width = image.shape

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

        theta = 2*pi*np.linspace(0,self.polar_width-1,self.polar_width)/self.polar_width
       
        pxCirclePoints = np.around(pupil_xyr[0] + pupil_xyr[2]*np.cos(theta))    
        ixCirclePoints = np.around(iris_xyr[0] + iris_xyr[2]*np.cos(theta))
        pyCirclePoints = np.around(pupil_xyr[1] + pupil_xyr[2]*np.sin(theta))
        iyCirclePoints = np.around(iris_xyr[1] + iris_xyr[2]*np.sin(theta))

        for j in range(1, self.polar_width+1):            
            for i in range(1, self.polar_height+1):

                radius = i/self.polar_height
                x = int(np.around((1-radius) * pxCirclePoints[j-1] + radius * ixCirclePoints[j-1]))
                y = int(np.around((1-radius) * pyCirclePoints[j-1] + radius * iyCirclePoints[j-1]))
                if (x > 0 and x <= width and y > 0 and y <= height): 
                    image_polar[i-1][j-1] = image[y-1][x-1]

        return image_polar

    @torch.inference_mode()
    def extractVector(self, polar):
        im_polar = Image.fromarray(polar, "L")
        im_tensor = self.input_transform(im_polar).unsqueeze(0).repeat(1,3,1,1).to(self.device)
        vector = self.nn_model(im_tensor)
        return vector.cpu().numpy()[0]
        
    @torch.inference_mode()
    def matchVectors(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        magnitude_a = np.linalg.norm(vector1)
        magnitude_b = np.linalg.norm(vector2)
        cosine_sim = dot_product / (magnitude_a * magnitude_b)
        return math.acos(cosine_sim)
        