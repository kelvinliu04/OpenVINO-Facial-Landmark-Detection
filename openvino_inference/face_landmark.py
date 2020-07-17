import os
import sys
import logging as log
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class FacialLandmark:
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name = model_name
        self.device = device
        self.extensions = extensions

    def load_model(self):
        ## Get model_bin and model_xml
        model_bin = self.model_name + ".bin"
        model_xml = self.model_name + ".xml"
        plugin = IECore()
        network = IENetwork(model=model_xml, weights=model_bin)
        ## Add extension if any
        if self.extensions and "CPU" in self.device:                # Add a CPU extension, if applicable
            plugin.add_extension(self.extensions, self.device)
        ## (Additional) Check unsupported layer 
        supported_layers = plugin.query_network(network=network, device_name=self.device)
        unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) > 2:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        ## Load network
        self.exec_network = plugin.load_network(network, self.device)                                        
        self.input_blob = next(iter(network.inputs))
        self.output_blob = next(iter(network.outputs))
        self.n, self.c, self.h, self.w = network.inputs[self.input_blob].shape
        self.plugin = plugin
        self.network = network
        
        
    def predict(self, image):
        h, w, _ = image.shape
        image = self.preprocess_input(image)
        self.exec_network.requests[0].infer({self.input_blob: image})
        #outputs = self.exec_network.requests[0].outputs['align_fc3']
        outputs = self.exec_network.requests[0].outputs[self.output_blob]
        landmarks = self.preprocess_output(outputs[0], h, w)
        return landmarks
    
    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        img = cv2.dnn.blobFromImage(image, size=(self.w, self.h))
        return img

    def preprocess_output(self, outputs, h, w):
        array = []
        for i in range(int(len(outputs)/2)):
            real_i = i*2
            array.append([int(outputs[real_i] * w) , int(outputs[real_i-1] * h)])
        return array
    
    def get_eyes(self, landmarks, face_img):
        '''
        static function
        '''
        h, w, _ = face_img.shape
        ## Left eye
        x1_l, y1_l = landmarks[1]
        x2_l, y2_l = landmarks[0]
        # get center left
        center_l = [int((x1_l + x2_l)*0.5), int((y1_l + y2_l)*0.5)]
        # get half distance left
        dist_l = int(np.linalg.norm((x1_l - x2_l) - (y1_l - y2_l)))
        half_dist_l = int(dist_l/2)
        # get points
        ymin_l = center_l[1] - int(half_dist_l*0.8)
        ymax_l = center_l[1] + int(half_dist_l*0.8)
        xmin_l = center_l[0] - half_dist_l
        xmax_l = center_l[0] + half_dist_l
        # crop left eye
        lEye = face_img[ymin_l:ymax_l, xmin_l:xmax_l]
        l_coords = (xmin_l, ymin_l), (xmax_l, ymax_l)
        
        ## Right eye
        x1_r, y1_r = landmarks[2]
        x2_r, y2_r = landmarks[3]
        # get center right
        center_r = [int((x1_r + x2_r)*0.5), int((y1_r + y2_r)*0.5)]
        # get half distance right
        dist_r = int(np.linalg.norm((x1_r - x2_r) - (y1_r - y2_r)))
        half_dist_r = int(dist_r/2)
        # get points
        ymin_r = center_r[1] - int(half_dist_l*0.8)
        ymax_r = center_r[1] + int(half_dist_l*0.8)
        xmin_r = center_r[0] - half_dist_l
        xmax_r = center_r[0] + half_dist_l
        # crop left eye
        rEye = face_img[ymin_r:ymax_r, xmin_r:xmax_r]
        r_coords = (xmin_r, ymin_r), (xmax_r, ymax_r)
        # coords (for visualization)
        coords = [l_coords, r_coords]

        return lEye, rEye, coords
        