import numpy as np
import timeout_decorator
import traceback
from utils.logger import Logger
from utils.tf_utils import norm_grad
from collections import defaultdict
import json

from grad_compress.grad_drop import GDropUpdate
from grad_compress.sign_sgd import SignSGDUpdate,MajorityVote
from comm_effi import StructuredUpdate

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

L = Logger()
logger = L.get_logger()

class Server:
    
    def __init__(self, client_model, clients=[], cfg=None,):
        self._cur_time = 0      # simulation time
        self.cfg = cfg
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.all_clients = clients
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []
        self.clients_info = defaultdict(dict)
        self.structure_updater = None
        for c in self.all_clients:
            self.clients_info[str(c.id)]["comp"] = 0
            self.clients_info[str(c.id)]["acc"] = 0.0
            self.clients_info[str(c.id)]["device"] = c.device.device_model
            self.clients_info[str(c.id)]["sample_num"] = len(c.train_data['y'])

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        if num_clients < self.cfg.min_selected:
            logger.info('insufficient clients: need {} while get {} online'.format(self.cfg.min_selected, num_clients))
            return False
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, deadline=-1):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            deadline: -1 for unlimited; >0 for each client's deadline
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0,
                   'acc': {},
                   'loss': {},
                   'ori_d_t': 0,
                   'ori_t_t': 0,
                   'ori_u_t': 0,
                   'act_d_t': 0,
                   'act_t_t': 0,
                   'act_u_t': 0} for c in clients}
        # for c in self.all_clients:
            # c.model.set_params(self.model)
        simulate_time = 0
        accs = []
        losses = []
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []
        for c in clients:
            c.model.set_params(self.model)
            try:
                # set deadline 
                c.set_deadline(deadline)
                # training
                logger.debug('client {} starts training...'.format(c.id))
                start_t = self.get_cur_time()
                # gradiant here is actually (-1) * grad
                simulate_time_c, comp, num_samples, update, acc, loss, gradiant, update_size, seed, shape_old, loss_old = c.train(start_t, num_epochs, batch_size, minibatch)       
                logger.debug('client {} simulate_time: {}'.format(c.id, simulate_time_c))
                logger.debug('client {} num_samples: {}'.format(c.id, num_samples))
                logger.debug('client {} acc: {}, loss: {}'.format(c.id, acc, loss))
                accs.append(acc)
                losses.append(loss)
                simulate_time = min(deadline, max(simulate_time, simulate_time_c))
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += update_size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                sys_metrics[c.id]['acc'] = acc
                sys_metrics[c.id]['loss'] = loss
                # uploading 
                self.updates.append((c.id, num_samples, update))

                if self.cfg.structure_k:
                    if not self.structure_updater:
                        self.structure_updater = StructuredUpdate(self.cfg.structure_k, seed)
                    gradiant = self.structure_updater.regain_grad(shape_old, gradiant)
                    # print("client {} finish using structure_update with k = {}".format(c.id, self.cfg.structure_k))
                    
                self.gradiants.append((c.id, num_samples, gradiant))

                if self.cfg.qffl:
                    q = self.cfg.qffl_q
                    # gradiant = [(-1) * grad for grad in gradiant]
                    self.deltas.append([np.float_power(loss_old + 1e-10, q) * grad for grad in gradiant])
                    self.hs.append(q * np.float_power(loss_old + 1e-10, (q - 1)) * norm_grad(gradiant) + (1.0 / self.client_model.lr) * np.float_power(loss_old + 1e-10, q))
                    # print("client {} finish using qffl with q = {}".format(c.id, self.cfg.qffl_q))
                
                norm_comp = int(comp/self.client_model.flops)
                if norm_comp == 0:
                    logger.error('comp: {}, flops: {}'.format(comp, self.client_model.flops))
                    assert False
                self.clients_info[str(c.id)]["comp"] += norm_comp
                # print('client {} upload successfully with acc {}, loss {}'.format(c.id,acc,loss))
                logger.debug('client {} upload successfully with acc {}, loss {}'.format(c.id,acc,loss))
            except timeout_decorator.timeout_decorator.TimeoutError as e:
                # logger.debug('client {} failed: {}'.format(c.id, e))
                # if "interruption" in str(e):
                logger.debug('client {} failed: {}'.format(c.id, e))
                actual_comp = c.get_actual_comp()
                norm_comp = int(actual_comp/self.client_model.flops)
                self.clients_info[str(c.id)]["comp"] += norm_comp
                
                sys_metrics[c.id]['acc'] = -1
                sys_metrics[c.id]['loss'] = -1
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.update_size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] += actual_comp
                
                simulate_time = deadline
            except Exception as e:
                logger.error('client {} failed: {}'.format(c.id, e))
                # logger.error('train_x: {}'.format(c.train_data['x']))
                # logger.error('train_y: {}'.format(c.train_data['y']))
                traceback.print_exc()
            finally:
                if self.cfg.compress_algo:
                    sys_metrics[c.id]['before_cprs_u_t'] = round(c.before_comp_upload_time, 3)
                sys_metrics[c.id]['ori_d_t'] = round(c.ori_download_time, 3)
                sys_metrics[c.id]['ori_t_t'] = round(c.ori_train_time, 3)
                sys_metrics[c.id]['ori_u_t'] = round(c.ori_upload_time, 3)
                
                sys_metrics[c.id]['act_d_t'] = round(c.act_download_time, 3)
                sys_metrics[c.id]['act_t_t'] = round(c.act_train_time, 3)
                sys_metrics[c.id]['act_u_t'] = round(c.act_upload_time, 3)
        try:
            # logger.info('simulation time: {}'.format(simulate_time))
            sys_metrics['configuration_time'] = simulate_time
            avg_acc = sum(accs)/len(accs)
            avg_loss = sum(losses)/len(losses)
            logger.info('average acc: {}, average loss: {}'.format(avg_acc, avg_loss))
            logger.info('configuration and update stage simulation time: {}'.format(simulate_time))
            # logger.info('losses: {}'.format(losses))
        except ZeroDivisionError as e:
            logger.error('training time window is too short to train!')
            # assert False
        except Exception as e:
            logger.error('failed reason: {}'.format(e))
            traceback.print_exc()
            assert False
        return sys_metrics

    def update_model(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:        
            logger.info('round succeed, updating global model...')
            if self.cfg.no_training:
                logger.info('pseduo-update because of no_training setting.')
                self.updates = []
                self.gradiants = []
                self.deltas = []
                self.hs = []
                return
            if self.cfg.aggregate_algorithm == 'FedAvg':
                # aggregate all the clients
                logger.info('Aggragate with FedAvg')
                used_client_ids = [cid for (cid, client_samples, client_model) in self.updates]
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_model) in self.updates:
                    total_weight += client_samples
                    for i, v in enumerate(client_model):
                        base[i] += (client_samples * v.astype(np.float64))
                for c in self.all_clients:
                    if c.id not in used_client_ids:
                        # c was not trained in this round
                        params = self.model
                        total_weight += c.num_train_samples  # assume that all train_data is used to update
                        for i, v in enumerate(params):
                            base[i] += (c.num_train_samples * v.astype(np.float64))
                averaged_soln = [v / total_weight for v in base]
                self.model = averaged_soln
            
            elif self.cfg.aggregate_algorithm == 'SucFedAvg':
                # aggregate the successfully uploaded clients
                logger.info('Aggragate with SucFedAvg')
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_model) in self.updates:
                    # logger.info('cid: {}, client_samples: {}, client_model: {}'.format(cid, client_samples, client_model[0][0][:5]))
                    total_weight += client_samples
                    for i, v in enumerate(client_model):
                        base[i] += (client_samples * v.astype(np.float64))
                averaged_soln = [v / total_weight for v in base]
                self.model = averaged_soln
            
            elif self.cfg.aggregate_algorithm == 'SelFedAvg':
                # aggregate the selected clients
                logger.info('Aggragate with SelFedAvg')
                used_client_ids = [cid for (cid, client_samples, client_model) in self.updates]
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_model) in self.updates:
                    total_weight += client_samples
                    for i, v in enumerate(client_model):
                        base[i] += (client_samples * v.astype(np.float64))
                for c in self.selected_clients:
                    if c.id not in used_client_ids:
                        # c was failed in this round but was selected
                        params = self.model
                        total_weight += c.num_train_samples  # assume that all train_data is used to update
                        for i, v in enumerate(params):
                            base[i] += (c.num_train_samples * v.astype(np.float64))
                averaged_soln = [v / total_weight for v in base]
                self.model = averaged_soln
            
            else:
                # not supported aggregating algorithm
                logger.error('not supported aggregating algorithm: {}'.format(self.cfg.aggregate_algorithm))
                assert False
                
        else:
            logger.info('round failed, global model maintained.')
        
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []

    def update_using_compressed_grad(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.gradiants) / len(self.selected_clients) >= update_frac:        
            logger.info('round succeed, updating global model...')
            if self.cfg.no_training:
                logger.info('pseduo-update because of no_training setting.')
                self.updates = []
                self.gradiants = []
                self.deltas = []
                self.hs = []
                return
            if self.cfg.aggregate_algorithm == 'FedAvg':
                # aggregate all the clients
                logger.info('Aggragate with FedAvg after grad compress')
                used_client_ids = [cid for (cid, _, _) in self.gradiants]
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_grad) in self.gradiants:
                    total_weight += client_samples
                    for i, v in enumerate(client_grad):
                        base[i] += (client_samples * v.astype(np.float32))
                for c in self.all_clients:
                    if c.id not in used_client_ids:
                        total_weight += c.num_train_samples
                averaged_grad = [v / total_weight for v in base]
                # update with grad                    
                if self.cfg.compress_algo == 'sign_sgd':
                    averaged_grad = MajorityVote(averaged_grad)
                self.model = self.client_model.update_with_gradiant(averaged_grad)                    

            elif self.cfg.aggregate_algorithm == 'SucFedAvg':
                # aggregate the successfully uploaded clients
                logger.info('Aggragate with SucFedAvg after grad compress')
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_grad) in self.gradiants:
                    # logger.info('cid: {}, client_samples: {}, client_model: {}'.format(cid, client_samples, client_model[0][0][:5]))
                    total_weight += client_samples
                    # logger.info("client-grad: {}, c-g[0]: {}".format(type(client_grad), type(client_grad[0])))
                    for i, v in enumerate(client_grad):
                        base[i] += (client_samples * v.astype(np.float32))
                averaged_grad = [v / total_weight for v in base]
                # update with grad                    
                if self.cfg.compress_algo == 'sign_sgd':
                    averaged_grad = MajorityVote(averaged_grad)
                self.model = self.client_model.update_with_gradiant(averaged_grad)
            
            elif self.cfg.aggregate_algorithm == 'SelFedAvg':
                # aggregate the selected clients
                logger.info('Aggragate with SelFedAvg after grad compress')
                used_client_ids = [cid for (cid, _, _) in self.gradiants]
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_grad) in self.gradiants:
                    total_weight += client_samples
                    for i, v in enumerate(client_grad):
                        base[i] += (client_samples * v.astype(np.float32))
                for c in self.selected_clients:
                    if c.id not in used_client_ids:
                        # c was failed in this round but was selected
                        total_weight += c.num_train_samples  # assume that all train_data is used to update
                averaged_grad = [v / total_weight for v in base]
                # update with grad                    
                if self.cfg.compress_algo == 'sign_sgd':
                    averaged_grad = MajorityVote(averaged_grad)
                self.model = self.client_model.update_with_gradiant(averaged_grad)
            
            else:
                # not supported aggregating algorithm
                logger.error('not supported aggregating algorithm: {}'.format(self.cfg.aggregate_algorithm))
                assert False
                
        else:
            logger.info('round failed, global model maintained.')
        
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []

    def update_using_qffl(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.gradiants) / len(self.selected_clients) >= update_frac:        
            logger.info('round succeed, updating global model...')
            if self.cfg.no_training:
                logger.info('pseduo-update because of no_training setting.')
                self.updates = []
                self.gradiants = []
                self.deltas = []
                self.hs = []
                return
            
            # aggregate using q-ffl
            demominator = np.sum(np.asarray(self.hs))
            num_clients = len(self.deltas)
            scaled_deltas = []
            for client_delta in self.deltas:
                scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

            updates = []
            for i in range(len(self.deltas[0])):
                tmp = scaled_deltas[0][i]
                for j in range(1, len(self.deltas)):
                    tmp += scaled_deltas[j][i]
                updates.append(tmp)

            self.model = [(u - v) * 1.0 for u, v in zip(self.model, updates)]

            self.updates = []
            self.gradiants = []
            self.deltas = []
            self.hs = []

        else:
            logger.info('round failed, global model maintained.')
        

    
    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients
            assert False

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            # logger.info('client {} metrics: {}'.format(client.id, c_metrics))
            metrics[client.id] = c_metrics
            if isinstance(c_metrics['accuracy'], np.ndarray):
                self.clients_info[client.id]['acc'] = c_metrics['accuracy'].tolist()
            else:
                self.clients_info[client.id]['acc'] = c_metrics['accuracy']
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.all_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess =  self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()
    
    def get_cur_time(self):
        return self._cur_time

    def pass_time(self, sec):
        self._cur_time += sec
    
    def get_time_window(self):
        tw =  np.random.normal(self.cfg.time_window[0], self.cfg.time_window[1])
        while tw < 0:
            tw =  np.random.normal(self.cfg.time_window[0], self.cfg.time_window[1])
        return tw
    
    def save_clients_info(self):
        with open('clients_info_{}.json'.format(self.cfg.config_name), 'w') as f:
            json.dump(self.clients_info, f)
        logger.info('save clients_info.json.')
