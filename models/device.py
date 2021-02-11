# simulate device type
# current classify as big/middle/small device
# device can also be 
from utils.logger import Logger
from utils.device_util.device_util import Device_Util
import numpy as np
import json
import random

# -1 - self define device, 0 - small, 1 - mid, 2 - big

L = Logger()
logger = L.get_logger()

class Device():
        
    du = Device_Util()
    speed_distri = None
    try:
        with open('speed_distri.json', 'r') as f:
            speed_distri = json.load(f) 
    except FileNotFoundError as e:
        speed_distri = None
        logger.warn('no user\'s network speed trace was found, set all communication time to 0.0s')


    # support device type
    def __init__(self, cfg, model_size = 0):
        self.device_model = None    # later set according to the trace
        self.cfg = cfg
        
        self.model_size = model_size / 1024 # change to kb because speed data use 'kb/s'
        if cfg.behav_hete == False and cfg.hard_hete == False:
            # make sure the no_trace mode perform the same as original leaf
            self.model_size = 0
        if Device.speed_distri == None:
            # treat as no_trace mode
            self.model_size = 0
            self.upload_speed_u = 1.0
            self.upload_speed_sigma = 0.0
            self.download_speed_u = 1.0
            self.download_speed_sigma = 0.0
        else:
            if cfg.hard_hete == False:
                # assign a fixed speed distribution
                guid = list(Device.speed_distri.keys())[cfg.seed%len(Device.speed_distri)]
                # logger.info(guid)
            else:
                guid = random.sample(list(Device.speed_distri.keys()), 1)[0]
            self.download_speed_u = Device.speed_distri[guid]['down_u']
            self.download_speed_sigma = Device.speed_distri[guid]['down_sigma']
            self.upload_speed_u = Device.speed_distri[guid]['up_u']
            self.upload_speed_sigma = Device.speed_distri[guid]['up_sigma']
            
        
        Device.du.set_model(cfg.model)
        Device.du.set_dataset(cfg.dataset)

    def set_device_model(self, real_device_model):
        self.device_model = Device.du.transfer(real_device_model)

    
    
    def get_upload_time(self, model_size):
        if self.model_size == 0.0 :
            return 0.0

        upload_speed = np.random.normal(self.upload_speed_u, self.upload_speed_sigma)
        while upload_speed < 0:
            upload_speed = np.random.normal(self.upload_speed_u, self.upload_speed_sigma)
        upload_time = model_size / upload_speed / 1000
        return float(upload_time)

    def get_download_time(self):
        if self.model_size == 0.0:            
            return 0.0
        
        download_speed = np.random.normal(self.download_speed_u, self.download_speed_sigma)
        while download_speed < 0:
            download_speed = np.random.normal(self.download_speed_u, self.download_speed_sigma)
        download_time = self.model_size / download_speed
        return float(download_time)
    
    def get_train_time(self, num_sample, batch_size, num_epoch):
        # TODO - finish train time predictor

        # current implementation: 
        # use real data withour prediction, 
        # so now it does not support other models
        if self.device_model == None:
            assert False
        return Device.du.get_train_time(self.device_model, num_sample, batch_size, num_epoch)
        