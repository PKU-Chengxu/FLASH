import numpy as np
import os
from grad_compress.compressor import Compressor
import json

# gradient_list: type  list
# gradient_list[0]: type  numpy.ndarray

temp_dir = 'grad_compress/temp/'

class GDropUpdate(Compressor):
    
    def __init__(self, client_id, cfg, threshold=0.005):
        self.delta_gradient_list_dir = temp_dir+'{}/'.format(cfg.config_name)
        self.delta_gradient_list_file = temp_dir+'{}/{}.json'.format(cfg.config_name, client_id)
        self.threshold = threshold
        if os.path.isdir(self.delta_gradient_list_dir):
            os.system('rm -r {}'.format(self.delta_gradient_list_dir))
        super(GDropUpdate, self).__init__()

    def GradientCompress(self, gradient_list):
        if os.path.exists(self.delta_gradient_list_file):
            with open(self.delta_gradient_list_file, 'r') as f:
                delta_gradient_list = json.load(f)
        else:
            os.makedirs(self.delta_gradient_list_dir, exist_ok=True)
            delta_gradient_list = []
            for i in range(len(gradient_list)):
                delta_gradient_list.append(gradient_list[i]*0)
        compressed_gradient_list = []
        element_num = 0
        compressed_element_num = 0
        for i in range(len(gradient_list)):
            gradient = gradient_list[i]
            element_num += np.size(gradient)
            gradient += delta_gradient_list[i]
            sign_gradient = np.sign((np.abs(gradient)/self.threshold).astype(np.int32))
            compressed_element_num += np.sum(sign_gradient)
            compressed_gradient = gradient*sign_gradient
            compressed_gradient_list.append(compressed_gradient)
            delta_gradient_list[i] = gradient - compressed_gradient

        with open(self.delta_gradient_list_file, 'w') as f:
            data = [_.tolist() for _ in  delta_gradient_list]
            json.dump(data, f)
        return compressed_gradient_list, element_num * 32,  compressed_element_num*32


# grad_drop_updater = GDropUpdate(0.005)

