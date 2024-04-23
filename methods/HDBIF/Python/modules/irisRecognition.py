import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import math
from math import pi
from torchvision import models
from modules.network import *

class irisRecognition(object):
    def __init__(self, cfg):
        # cParsing the config file
        self.polar_height = cfg["polar_height"]
        self.polar_width = cfg["polar_width"]
        self.filter_size = cfg["recog_filter_size"]
        self.num_filters = cfg["recog_num_filters"]
        self.max_shift = cfg["recog_max_shift"]
        self.cuda = cfg["cuda"]
        self.threshold_frac_avg_bits = cfg["threshold_frac_avg_bits"]
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.mask_model_path = cfg["mask_model_path"]
        self.circle_model_path = cfg["circle_model_path"]
        mat_file_path = cfg["recog_bsif_dir"]+'ICAtextureFilters_{0}x{1}_{2}bit.pt'.format(self.filter_size, self.filter_size, self.num_filters)
        with torch.inference_mode():
            filter_mat = torch.jit.load(mat_file_path, torch.device('cpu')).ICAtextureFilters.detach().numpy()
        #filter_mat_scipy = scipy.io.loadmat('../filters/finetuned_bsif_eyetracker_data/'+'ICAtextureFilters_{0}x{1}_{2}bit.mat'.format(self.filter_size, self.filter_size, self.num_filters))['ICAtextureFilters']
        self.filter_size = filter_mat.shape[0]
        self.num_filters = filter_mat.shape[2]
        with torch.inference_mode():
            self.torch_filter = torch.FloatTensor(filter_mat).to(self.device)
            self.torch_filter = torch.moveaxis(self.torch_filter.unsqueeze(0), 3, 0).detach().requires_grad_(False)
        
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
            self.circle_model = self.circle_model.to(self.device)
            self.circle_model.eval()
            self.mask_model = NestedSharedAtrousResUNet(1, 1, width=32, resolution=(240,320))
            try:
                self.mask_model.load_state_dict(torch.load(self.mask_model_path, map_location=self.device))
            except AssertionError:
                    print("assertion error")
                    self.mask_model.load_state_dict(torch.load(self.mask_model_path,
                        map_location = lambda storage, loc: storage))
            self.mask_model = self.mask_model.to(self.device)
            self.mask_model.eval()
            self.input_transform_mask = Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])
            self.input_transform_circ = Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])          
        
        avg_bits_by_filter_size = {5: 25056, 7: 24463, 9: 23764, 11: 23010, 13: 22225, 15: 21420, 17: 20603, 19: 19777, 21: 18945, 27: 16419, 33: 13864, 39: 11289}
        self.avg_num_bits = avg_bits_by_filter_size[self.filter_size]
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
    
    def segment_and_circApprox(self, image):
        pred = self.segment(image)
        pupil_xyr, iris_xyr = self.circApprox(pred, image)
        return pred, pupil_xyr, iris_xyr

    @torch.inference_mode()
    def segment(self,image):

        w,h = image.size
        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        mask_logit_t = self.mask_model(Variable(self.input_transform_mask(image).unsqueeze(0).to(self.device)))[0]
        mask_t = torch.where(torch.sigmoid(mask_logit_t) > 0.5, 255, 0)
        mask = mask_t.cpu().numpy()[0]
        mask = cv2.resize(np.uint8(mask), (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        #print('Mask Shape: ', mask.shape)

        return mask

    def segmentVis(self,im,mask,pupil_xyr,iris_xyr):
        
        pupil_xyr = np.around(pupil_xyr).astype(np.int32)
        iris_xyr = np.around(iris_xyr).astype(np.int32)
        imVis = np.stack((np.array(im),)*3, axis=-1)
        imVis[:,:,1] = np.clip(imVis[:,:,1] + (96/255)*mask,0,255)
        imVis = cv2.circle(imVis, (pupil_xyr[0],pupil_xyr[1]), pupil_xyr[2], (0, 0, 255), 2)
        imVis = cv2.circle(imVis, (iris_xyr[0],iris_xyr[1]), iris_xyr[2], (255, 0, 0), 2)

        return imVis

    @torch.inference_mode()
    def circApprox(self,mask=None,image=None):

        w,h = image.size

        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        inp_xyr_t = self.circle_model(Variable(self.input_transform_circ(image).unsqueeze(0).repeat(1,3,1,1).to(self.device)))

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
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, align_corners=True)

    @torch.inference_mode()
    def cartToPol_torch(self, image, mask, pupil_xyr, iris_xyr, interpolation='bilinear'): # 
        if pupil_xyr is None or iris_xyr is None:
            return None, None
        
        image = torch.tensor(np.array(image)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask = torch.tensor(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to(self.device)
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
        mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest') # always use nearest neighbor interpolation for mask
        mask_polar = (mask_polar>0.5).long() * 255

        return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)

    # (Fixed) Old implementation of Rubbersheet model-based Cartesian-to-polar transformation that uses nearest neighbor interpolation
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        
        if pupil_xyr is None:
            return None, None
       
        image = np.array(image)
        height, width = image.shape
        mask = np.array(mask)

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)
        mask_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

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
                    mask_polar[i-1][j-1] = mask[y-1][x-1]

        return image_polar, mask_polar

    @torch.inference_mode()
    def extractCode(self, polar):
        if polar is None:
            return None
        r = int(np.floor(self.filter_size / 2))
        polar_t = torch.tensor(polar).float().unsqueeze(0).unsqueeze(0).to(self.device)
        #polar_t = (polar_t - polar_t.min()) / (polar_t.max() - polar_t.min())
        padded_polar = nn.functional.pad(polar_t, (r, r, 0, 0), mode='circular')
        codeContinuous = nn.functional.conv2d(padded_polar, self.torch_filter)
        codeBinary = torch.where(codeContinuous.squeeze(0) > 0, True, False).cpu().numpy()
        return codeBinary # The size of the code should be: 7 x (64 - filter_size) x 512 
    
    @torch.inference_mode()
    def extractMultipleCodes(self, polars):
        polars_t = []
        for polar in polars:
            if polar is None:
                print("One of the polar images is None. Unable to continue...")
                return None
            r = int(np.floor(self.filter_size / 2))
            polar_t = torch.tensor(polar).float().unsqueeze(0).unsqueeze(0).to(self.device)
            polars_t.append(polar_t)
        polars_t = torch.cat(polars_t, 0)
        padded_polar = nn.functional.pad(polars_t, (r, r, 0, 0), mode='circular')
        codesContinuous = nn.functional.conv2d(padded_polar, self.torch_filter)
        codesBinary = torch.where(codesContinuous > 0, True, False).cpu().numpy()
        codes = []
        for i in range(codesBinary.shape[0]):
            codes.append(codesBinary[i]) # The size of the code should be: 7 x (64 - filter_size) x 512
        return codes

    def matchCodes(self, code1, code2, mask1, mask2):
        r = int(np.floor(self.filter_size / 2))
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1[r:-r, :] > 127, True, False)
        mask2_binary = np.where(mask2[r:-r, :] > 127, True, False)
        if (np.sum(mask1_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits) or (np.sum(mask2_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits):
            print("Too small masks")
            return -1.0, -1.0
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.num_filters))
            if scoreC[-1] != float('inf'):
                scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( np.sum(andMasks) / self.avg_num_bits )
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]
        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
                
        return scoreC_best, scoreC_index - self.max_shift

    def matchCodesEfficient(self, code1, code2, mask1, mask2):
        r = int(np.floor(self.filter_size / 2))
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1[r:-r, :] > 127, True, False)
        mask2_binary = np.where(mask2[r:-r, :] > 127, True, False)
        if (np.sum(mask1_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits) or (np.sum(mask2_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits):
            print("Too small masks")
            return -1.0, -1.0
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1, 2):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.num_filters))
            if scoreC[-1] != float('inf'):
                scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( np.sum(andMasks) / self.avg_num_bits )
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_shift = scoreC_index * 2 - self.max_shift
        scoreC = scoreC[scoreC_index]
        if scoreC == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
        
        andMasks_left = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift-1, axis=1))
        if np.sum(andMasks_left) == 0:
            scoreC_left = float('inf')
        else:
            xorCodes_left = np.logical_xor(code1, np.roll(code2, scoreC_shift-1, axis=2))
            xorCodesMasked_left = np.logical_and(xorCodes_left, np.tile(np.expand_dims(andMasks_left,axis=0), (self.num_filters, 1, 1)))
            scoreC_left = np.sum(xorCodesMasked_left) / (np.sum(andMasks_left) * self.num_filters)
        
        andMasks_right = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift+1, axis=1))
        if np.sum(andMasks_right) == 0:
            scoreC_right = float('inf')
        else:
            xorCodes_right = np.logical_xor(code1, np.roll(code2, scoreC_shift+1, axis=2))
            xorCodesMasked_right = np.logical_and(xorCodes_right, np.tile(np.expand_dims(andMasks_right,axis=0), (self.num_filters, 1, 1)))
            scoreC_right = np.sum(xorCodesMasked_right) / (np.sum(andMasks_right) * self.num_filters)
        
        scoreC_best = min(scoreC, scoreC_left, scoreC_right)
        if scoreC_best == scoreC_left:
            scoreC_shift -= 1
        elif scoreC_best == scoreC_right:
            scoreC_shift += 1
        
        return scoreC_best, scoreC_shift