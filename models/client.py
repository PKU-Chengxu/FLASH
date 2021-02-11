import random
import warnings
import timeout_decorator
import sys
import numpy as np
import json

from utils.logger import Logger
from device import Device
from timer import Timer

from grad_compress.grad_drop import GDropUpdate
from grad_compress.sign_sgd import SignSGDUpdate
from comm_effi import StructuredUpdate

L = Logger()
logger = L.get_logger()

class Client:
    
    d = None
    try:
        with open('../data/state_traces.json', 'r', encoding='utf-8') as f:
            d = json.load(f)
    except FileNotFoundError as e:
        d = None
        logger.warn('no user behavior trace was found, running in no-trace mode')
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None, device=None, cfg=None):
        self._model = model
        self.id = client_id # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.deadline = 1 # < 0 for unlimited
        self.cfg = cfg
        
        self.compressor = None
        if self.cfg.compress_algo:
            if self.cfg.compress_algo == 'sign_sgd':
                self.compressor = SignSGDUpdate()
            elif self.cfg.compress_algo == 'grad_drop':
                self.compressor = GDropUpdate(client_id,cfg)
            else:
                logger.error("compress algorithm is not defined")
        
        self.structured_updater = None
        if self.cfg.structure_k:
            self.structured_updater = StructuredUpdate(self.cfg.structure_k, self.cfg.seed)
        
        self.device = device  # if device == none, it will use real time as train time, and set upload/download time as 0
        if self.device == None:
            logger.warn('client {} with no device init, upload time will be set as 0 and speed will be the gpu speed'.format(self.id))
            self.upload_time = 0
        
        # timer
        d = Client.d
        if d == None:
            cfg.behav_hete = False
        # uid = random.randint(0, len(d))
        if cfg.behav_hete:
            if cfg.real_world == False:
                uid = random.sample(list(d.keys()), 1)[0]
                self.timer = Timer(ubt=d[str(uid)], google=True)
                while self.timer.isSuccess != True:
                    uid = random.sample(list(d.keys()), 1)[0]
                    self.timer = Timer(ubt=d[str(uid)], google=True)
            else:
                uid = self.id
                self.timer = Timer(ubt=d[str(uid)], google=True)
        else:
            # no behavior heterogeneity, always available
            self.timer = Timer(None)
            self.deadline = sys.maxsize # deadline is meaningless without user trace
        
        real_device_model = self.timer.model
        
        if not self.device: 
            self.device = Device(cfg, 0.0)

        if self.cfg.hard_hete:
            self.device.set_device_model(real_device_model)
        else:
            self.device.set_device_model("Redmi Note 8")


    def train(self, start_t=None, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            start_t: strat time of the training, only used in train_with_simulate_time
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            acc, loss, grad, update_size
        """
        
            
        def train_with_simulate_time(self, start_t, num_epochs=1, batch_size=10, minibatch=None):
            if minibatch is None:
                num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
            else :
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
            
            train_time = self.device.get_train_time(num_data, batch_size, num_epochs)
            logger.debug('client {}: num data:{}'.format(self.id, num_data))
            logger.debug('client {}: train time:{}'.format(self.id, train_time))
            
            # compute num_data
            if minibatch is None:
                num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}

            download_time = self.device.get_download_time()
            upload_time = self.device.get_upload_time(self.model.size) # will be re-calculated after training
            
            down_end_time = self.timer.get_future_time(start_t, download_time)
            logger.debug("client {} download-time-need={}, download-time-cost={} end at {}, "
                        .format(self.id, download_time, down_end_time-start_t, down_end_time))

            train_end_time = self.timer.get_future_time(down_end_time, train_time)
            logger.debug("client {} train-time-need={}, train-time-cost={} end at {}, "
                        .format(self.id, train_time, train_end_time-down_end_time, train_end_time))
            
            up_end_time = self.timer.get_future_time(train_end_time, upload_time)
            logger.debug("client {} upload-time-need={}, upload-time-cost={} end at {}, "
                        .format(self.id, upload_time, up_end_time-train_end_time, up_end_time))
            
            # total_cost = up_end_time - start_t
            # logger.debug("client {} task-time-need={}, task-time-cost={}"
            #             .format(self.id, download_time+train_time+upload_time, total_cost))

            self.ori_download_time = download_time  # original
            self.ori_train_time = train_time
            self.before_comp_upload_time = upload_time
            self.ori_upload_time = upload_time

            self.act_download_time = down_end_time-start_t # actual
            self.act_train_time = train_end_time-down_end_time
            self.act_upload_time = up_end_time-train_end_time   # maybe decrease for the use of conpression algorithm
            
            self.update_size = self.model.size

            '''
            if not self.timer.check_comm_suc(start_t, download_time):
                self.actual_comp = 0.0
                download_available_time = self.timer.get_available_time(start_t, download_time)
                failed_reason = 'download interruption: download_time({}) > download_available_time({})'.format(download_time, download_available_time)
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            if train_time > train_time_limit:
                # data sampling
                comp = self.model.get_comp(data, num_epochs, batch_size)
                self.actual_comp = int(comp*available_time/train_time)    # will be used in get_actual_comp
                failed_reason = 'out of deadline: download_time({}) + train_time({}) + upload_time({}) > deadline({})'.format(download_time, train_time, upload_time, self.deadline)
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            elif train_time > available_time:
                # client interruption
                comp = self.model.get_comp(data, num_epochs, batch_size)
                self.actual_comp = int(comp*available_time/train_time)    # will be used in get_actual_comp
                failed_reason = 'client interruption: train_time({}) > available_time({})'.format(train_time, available_time)
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            if not self.timer.check_comm_suc(start_t + download_time + train_time, upload_time):
                comp = self.model.get_comp(data, num_epochs, batch_size)
                self.actual_comp = comp
                upload_available_time = self.timer.get_available_time(start_t + download_time + train_time, upload_time)
                failed_reason = 'upload interruption: upload_time({}) > upload_available_time({})'.format(upload_time, upload_available_time)
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            '''
            if (down_end_time-start_t) > self.deadline:
                # download too long
                self.actual_comp = 0.0
                self.update_size = 0
                failed_reason = 'failed when downloading'
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            elif (train_end_time-start_t) > self.deadline:
                # failed when training
                train_time_limit = self.deadline - self.act_download_time
                if train_time_limit <= 0:
                    train_time_limit = 0.001
                available_time = self.timer.get_available_time(start_t + self.act_download_time, train_time_limit)
                comp = self.model.get_comp(data, num_epochs, batch_size)
                self.actual_comp = int(comp*available_time/train_time)    # will be used in get_actual_comp
                self.update_size = 0
                if self.cfg.fedprox:
                    ne = -1
                    for i in range(1, num_epochs):
                        et = self.timer.get_future_time(down_end_time, train_time*ne/num_epochs + upload_time)
                        if et - start_t <= self.deadline:
                            ne = i
                    if self.cfg.no_training:
                        comp = self.model.get_comp(data, num_epochs, batch_size)
                        update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                    elif self.cfg.fedprox and ne != -1:
                        comp, update, acc, loss, grad, loss_old = self.model.train(data, ne, batch_size)
                        train_time *= ne / num_epochs
                    else:
                        failed_reason = 'failed when training'
                        raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                else:
                    failed_reason = 'failed when training'
                    raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            elif (up_end_time-start_t) > self.deadline:
                self.actual_comp = self.model.get_comp(data, num_epochs, batch_size)
                if self.cfg.fedprox:
                    ne = -1
                    for i in range(1, num_epochs):
                        et = self.timer.get_future_time(down_end_time, train_time*ne/num_epochs + upload_time)
                        if et - start_t <= self.deadline:
                            ne = i
                    if self.cfg.no_training:
                        comp = self.model.get_comp(data, num_epochs, batch_size)
                        update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                    elif self.cfg.fedprox and ne != -1:
                        comp, update, acc, loss, grad, loss_old = self.model.train(data, ne, batch_size)
                        train_time *= ne / num_epochs
                    else:
                        failed_reason = 'failed when uploading'
                        raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                else:
                    failed_reason = 'failed when uploading'
                    raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            else :
                if minibatch is None:
                    if self.cfg.no_training:
                        comp = self.model.get_comp(data, num_epochs, batch_size)
                        update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                    else:
                        comp, update, acc, loss, grad, loss_old = self.model.train(data, num_epochs, batch_size)
                else:
                    # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                    num_epochs = 1
                    if self.cfg.no_training:
                        comp = self.model.get_comp(data, num_epochs, num_data)
                        update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                    else:
                        comp, update, acc, loss, grad, loss_old = self.model.train(data, num_epochs, batch_size)
            num_train_samples = len(data['y'])
            simulate_time_c = train_time + upload_time
            self.actual_comp = comp

            # gradiant compress and Federated Learning Strategies are mutually-exclusive
            # gradiant compress
            if self.compressor != None and not self.cfg.no_training:
                grad, size_old, size_new = self.compressor.GradientCompress(grad)
                # logger.info('compression ratio: {}'.format(size_new/size_old))
                self.update_size = self.update_size*size_new/size_old
                # re-calculate upload_time
                upload_time = self.device.get_upload_time(self.update_size)
                self.ori_upload_time = upload_time
                up_end_time = self.timer.get_future_time(train_end_time, upload_time)
                self.act_upload_time = up_end_time-train_end_time

            # Federated Learning Strategies for Improving Communication Efficiency
            seed = None
            shape_old = None
            if self.structured_updater and not self.cfg.no_training:
                seed, shape_old, grad = self.structured_updater.struc_update(grad)
                # logger.info('compression ratio: {}'.format(sum([np.prod(g.shape) for g in grad]) / sum([np.prod(s) for s in shape_old])))
                self.update_size *= sum([np.prod(g.shape) for g in grad]) / sum([np.prod(s) for s in shape_old])
                # re-calculate upload_time
                upload_time = self.device.get_upload_time(self.update_size)
                self.ori_upload_time = upload_time
                up_end_time = self.timer.get_future_time(train_end_time, upload_time)
                self.act_upload_time = up_end_time-train_end_time
            
            total_cost = self.act_download_time + self.act_train_time + self.act_upload_time
            if total_cost > self.deadline:
                # failed when uploading
                self.actual_comp = self.model.get_comp(data, num_epochs, batch_size)
                failed_reason = 'failed when uploading'
                # Note that, to simplify, we did not change the update_size here, actually the actual update size is less.
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            # if self.cfg.fedprox:
            #     print("client {} finish train task".format(self.id))
            return simulate_time_c, comp, num_train_samples, update, acc, loss, grad, self.update_size, seed, shape_old, loss_old
        '''
        # Deprecated
        @timeout_decorator.timeout(train_time_limit)
        def train_with_real_time_limit(self, num_epochs=1, batch_size=10, minibatch=None):
            logger.warn('call train_with_real_time_limit()')
            start_time = time.time()
            if minibatch is None:
                # data = self.train_data
                num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}
                if self.cfg.no_training:
                    comp, update, acc, loss, grad = -1,-1,-1,-1,-1
                else:
                    comp, update, acc, loss, grad = self.model.train(data, num_epochs, batch_size)
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}

                # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                num_epochs = 1
                if self.cfg.no_training:
                    comp, update, acc, loss, grad = -1,-1,-1,-1,-1
                else:
                    comp, update, acc, loss, grad = self.model.train(data, num_epochs, num_data)
            num_train_samples = len(data['y'])
            simulate_time_c = time.time() - start_time

            self.ori_download_time = 0  # original
            self.ori_train_time = simulate_time_c
            self.ori_upload_time = 0

            self.act_download_time = 0 # actual
            self.act_train_time = simulate_time_c
            self.act_upload_time = 0
            
            # gradiant compress
            update_size = self.model.size
            if grad != -1 and self.cfg.compress_algo:
                if self.cfg.compress_algo == 'sign_sgd':
                    grad, size_old, size_new = sign_sgd_updater.GradientCompress(grad)
                    update_size = update_size*size_new/size_old
                elif self.cfg.compress_algo == 'grad_drop':
                    grad, size_old, size_new = grad_drop_updater.GradientCompress(grad)
                    update_size = update_size*size_new/size_old
                else:
                    logger.error("compress algorithm is not defined")

            return simulate_time_c, comp, num_train_samples, update, acc, loss, grad, update_size
        '''

        return train_with_simulate_time(self, start_t, num_epochs, batch_size, minibatch)

        '''
        if self.device == None:
            return train_with_real_time_limit(self, num_epochs, batch_size, minibatch)
        else:
            return train_with_simulate_time(self, start_t, num_epochs, batch_size, minibatch)
        '''

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
    
    
    def set_deadline(self, deadline = -1):
        if deadline < 0 or not self.cfg.behav_hete:
            self.deadline = sys.maxsize
        else:
            self.deadline = deadline
        logger.debug('client {}\'s deadline is set to {}'.format(self.id, self.deadline))
    
    '''
    def set_upload_time(self, upload_time):
        if upload_time > 0:
            self.upload_time = upload_time
        else:
            logger.error('invalid upload time: {}'.format(upload_time))
            assert False
        logger.debug('client {}\'s upload_time is set to {}'.format(self.id, self.upload_time))
    
    def get_train_time_limit(self):
        if self.device != None:
            self.upload_time = self.device.get_upload_time()
            logger.debug('client {} upload time: {}'.format(self.id, self.upload_time))
        if self.upload_time < self.deadline :
            # logger.info('deadline: {}'.format(self.deadline))
            return self.deadline - self.upload_time
        else:
            return 0.01
    '''
    

    def upload_suc(self, start_t, num_epochs=1, batch_size=10, minibatch=None):
        """Test if this client will upload successfully

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            start_t: strat time of the training, only used in train_with_simulate_time
        Return:
            result: test result(True or False)
        """
        if minibatch is None:
            num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
        else :
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
        if self.device == None:
            download_time = 0.0
            upload_time = 0.0
        else:
            download_time = self.device.get_download_time()
            upload_time = self.device.get_upload_time()
        train_time = self.device.get_train_time(num_data, batch_size, num_epochs)
        train_time_limit = self.deadline - download_time - upload_time
        if train_time_limit < 0:
            train_time_limit = 0.001
        available_time = self.timer.get_available_time(start_t + download_time, train_time_limit)
        
        logger.debug('client {}: train time:{}'.format(self.id, train_time))
        logger.debug('client {}: available time:{}'.format(self.id, available_time))
        
        # compute num_data
        if minibatch is None:
            num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
        
        if not self.timer.check_comm_suc(start_t, download_time):
            return False
        if train_time > train_time_limit:
            return False
        elif train_time > available_time:
            return False
        if not self.timer.check_comm_suc(start_t + download_time + train_time, upload_time):
            return False
        else :
            return True

    
    def get_device_model(self):
        if self.device == None:
            return 'None'
        return self.device.device_model
        
    def get_actual_comp(self):
        '''
        get the actual computation in the training process
        '''
        return self.actual_comp
