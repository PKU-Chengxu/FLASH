import os
from .logger import Logger
import traceback
import sys

L = Logger()
logger = L.get_logger()

DEFAULT_CONFIG_FILE = 'default.cfg'

# configuration for FedAvg
class Config():
    def __init__(self, config_file = 'default.cfg'):
        self.config_name = config_file
        self.dataset = 'reddit'
        self.model = 'stacked_lstm'
        self.num_rounds = -1            # -1 for unlimited
        self.lr = 0.1
        self.eval_every = 3             # -1 for eval when quit
        self.clients_per_round = 10
        self.min_selected = 10
        self.max_sample = 100           # max sample num for training in a round
        self.batch_size = 10
        self.seed = 0
        self.metrics_file = 'metrics'
        self.num_epochs = 1
        self.gpu_fraction = 0.2
        self.minibatch = None       # always None for FedAvg
        self.round_ddl = [1000, 0]
        self.update_frac = 0.5
        self.max_client_num = 1000    # total client num, -1 for unlimited
        '''
        self.big_upload_time = [5.0, 1.0]
        self.mid_upload_time = [10.0, 1.0]
        self.small_upload_time = [15.0, 1.0]
        '''
        self.upload_time = [10.0, 1.0]  # now its no use in config
        '''
        # speed is no more used and replaced by training time provided by device_util
        self.big_speed = [150.0, 1.0]
        self.mid_speed = [100.0, 1.0]
        self.small_speed = [50.0, 1.0]  
        '''
        self.aggregate_algorithm = 'FedAvg'
        self.time_window = [20.0, 0.0]  # time window for selection stage
        self.user_trace = False
        self.behav_hete = False
        self.hard_hete = False
        self.no_training = False
        self.real_world = False
        # grad_compress,  structure_k, fedprox and qffl are mutually-exclusive
        self.compress_algo = None
        self.fedprox = False
        self.fedprox_mu = 0
        self.structure_k = None 
        self.qffl = False
        self.qffl_q = 0
        
        logger.info('read config from {}'.format(config_file))
        self.read_config(config_file)
        self.log_config()
        
        
    def read_config(self, filename = DEFAULT_CONFIG_FILE):
        if not os.path.exists(filename):
            logger.error('ERROR: config file {} does not exist!'.format(filename))
            assert False
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    line = line.strip().split()
                    if line[0] == 'num_rounds':
                        self.num_rounds = int(line[1])
                    elif line[0] == 'learning_rate':
                        self.lr = float(line[1])
                    elif line[0] == 'eval_every':
                        self.eval_every = int(line[1])
                    elif line[0] == 'clients_per_round':
                        self.clients_per_round = int(line[1])
                    elif line[0] == 'max_client_num':
                        self.max_client_num = int(line[1])
                        if self.max_client_num < 0:
                            self.max_client_num = sys.maxsize
                    elif line[0] == 'min_selected':
                        self.min_selected = int(line[1])
                    elif line[0] == 'batch_size':
                        self.batch_size = int(line[1])
                    elif line[0] == 'seed':
                        self.seed = int(line[1])
                    elif line[0] == 'metrics_file':
                        self.metrics_file = str(line[1])
                    elif line[0] == 'num_epochs':
                        self.num_epochs = int(line[1])
                    elif line[0] == 'dataset':
                        self.dataset = str(line[1])
                    elif line[0] == 'model':
                        self.model = str(line[1])
                    elif line[0] == 'gpu_fraction':
                        self.gpu_fraction = float(line[1])
                    elif line[0] == 'round_ddl':
                        self.round_ddl = [float(line[1]), float(line[2])]
                    elif line[0] == 'update_frac':
                        self.update_frac = float(line[1])
                    elif line[0] == 'upload_time':
                        self.upload_time = [float(line[1]), float(line[2])]
                    elif line[0] == 'aggregate_algorithm':
                        self.aggregate_algorithm = str(line[1])
                    elif line[0] == 'time_window':
                        self.time_window = [float(line[1]), float(line[2])]
                    elif line[0] == 'behav_hete' :
                        self.behav_hete = line[1].strip() == 'True'
                        if not self.behav_hete:
                            logger.info('no behavior heterogeneity! assume client is availiable at any time.')
                    elif line[0] == 'hard_hete' :
                        self.hard_hete = line[1].strip() == 'True'
                        if not self.hard_hete:
                            logger.info('no hardware heterogeneity! assume all clients are same.')
                    elif line[0] == 'no_training' :
                        self.no_training = line[1].strip() == 'True'
                        if self.no_training:
                            logger.info('no actual training process')
                    elif line[0] == 'realworld':
                        self.real_world = line[1].strip() == 'True'
                    elif line[0] == 'max_sample' :
                        self.max_sample = int(line[1])
                    elif line[0] == 'compress_algo':
                        self.compress_algo = line[1].strip()
                    elif line[0] == 'fedprox':
                        self.fedprox = line[1].strip()=='True'
                    elif line[0] == 'fedprox_mu':
                        self.fedprox_mu = float(line[1].strip())
                    elif line[0] == 'fedprox_active_frac':
                        self.fedprox_active_frac = float(line[1].strip())
                    elif line[0] == 'structure_k':
                        self.structure_k = int(line[1].strip())
                    elif line[0] == 'qffl':
                        self.qffl = line[1].strip()=='True'
                    elif line[0] == 'qffl_q':
                        self.qffl_q = float(line[1].strip())
                    elif line[0] == 'user_trace':
                        # to be compatibale with old version
                        self.user_trace = line[1].strip()=='True'
                except Exception as e:
                    traceback.print_exc()
        if self.real_world and 'realworld' not in self.dataset:
            logger.error('\'real_world\' is valid only when dataset is set to \'realworld\', current dataset {}'.format(self.dataset))
            self.real_world = False
        if self.user_trace == True:
            self.hard_hete = True
            self.behav_hete = True
    
    def log_config(self):
        configs = vars(self)
        logger.info('================= Config =================')
        for key in configs.keys():
            logger.info('\t{} = {}'.format(key, configs[key]))
        logger.info('================= ====== =================')
        
