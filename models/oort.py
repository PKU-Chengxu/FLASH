from collections import defaultdict
import math
import numpy as np
import random
from random import Random
from client import Client

from utils.logger import Logger
logger = Logger().get_logger()

class ClientSampler:
    def __init__(self, cfg = None, seed = 233) -> None:
        self.clients = {}
        self.training_round = 1
        self.unexplored = set()
        self.successfulClients = set()
        self.totalArms = defaultdict(dict)
        self.cfg = cfg
        self.exploration = cfg.oort_exploration_factor
        self.decay_factor = cfg.oort_exploration_decay
        self.exploration_min = cfg.oort_exploration_min
        self.alpha = cfg.oort_exploration_alpha
        self.sample_window = cfg.oort_sample_window
        self.last_util_record = 0
        self.round_threshold = cfg.oort_round_threshold

        self.rdm = Random()
        self.rdm.seed(seed)
        np.random.seed(seed)

        # seems not matter
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []



    def registerClient(self, clientId, feedback):
        self.totalArms[clientId] = {}
        self.totalArms[clientId]['reward'] = feedback['reward']
        self.totalArms[clientId]['duration'] = feedback['duration']
        self.totalArms[clientId]['time_stamp'] = self.training_round
        self.totalArms[clientId]['count'] = 1

        self.unexplored.add(clientId)
    
    def updateScore(self, clientId, reward, duration, time_stamp):
        self.totalArms[clientId]['reward'] = reward
        self.totalArms[clientId]['duration'] = duration
        self.totalArms[clientId]['time_stamp'] = time_stamp
        self.totalArms[clientId]['count'] += 1
        self.totalArms[clientId]['status'] = True

        self.unexplored.discard(clientId)
        self.successfulClients.add(clientId)
    

    def calculateSumUtil(self, clientList):
        cnt, cntUtil = 1e-4, 0

        for client in clientList:
            if client in self.successfulClients:
                cnt += 1
                cntUtil += self.totalArms[client]['reward']

        return cntUtil/cnt
    
    def pacer(self):
        # summarize utility in last epoch
        lastExplorationUtil = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)

        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)

        self.successfulClients = set()

        if self.training_round >= 2 * self.cfg.oort_pacer_step and self.training_round % self.cfg.oort_pacer_step == 0:

            utilLastPacerRounds = sum(self.exploitUtilHistory[-2 * self.cfg.oort_pacer_step : -self.cfg.oort_pacer_step])
            utilCurrentPacerRounds = sum(self.exploitUtilHistory[-self.cfg.oort_pacer_step:])

            # Cumulated statistical utility becomes flat, so we need a bump by relaxing the pacer
            if abs(utilCurrentPacerRounds - utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                self.round_threshold = min(100., self.round_threshold + self.cfg.oort_pacer_delta)
                self.last_util_record = self.training_round - self.cfg.oort_pacer_step
                logger.info("Training selector: Pacer changes at {} to {}".format(self.training_round, self.round_threshold))

            # change sharply -> we decrease the pacer step
            elif abs(utilCurrentPacerRounds - utilLastPacerRounds) >= utilLastPacerRounds * 5:
                self.round_threshold = max(self.cfg.oort_pacer_delta, self.round_threshold - self.cfg.oort_pacer_delta)
                self.last_util_record = self.training_round - self.cfg.oort_pacer_step
                logger.info("Training selector: Pacer changes at {} to {}".format(self.training_round, self.round_threshold))

            logger.info("Training selector: utilLastPacerRounds {}, utilCurrentPacerRounds {} in round {}"
                .format(utilLastPacerRounds, utilCurrentPacerRounds, self.training_round))

        logger.info("Training selector: Pacer {}: lastExploitationUtil {}, lastExplorationUtil {}, last_util_record {}".
                        format(self.training_round, lastExploitationUtil, lastExplorationUtil, self.last_util_record))
    

    def get_blacklist(self):
        # default return []
        blacklist = []

        if self.cfg.oort_blacklist_rounds != -1:
            sorted_client_ids = sorted(list(self.totalArms), reverse=True, key=lambda k: self.totalArms[k]['count'])

            for clientId in sorted_client_ids:
                if self.totalArms[clientId]['count'] > self.cfg.oort_blacklist_rounds:
                    blacklist.append(clientId)
                else:
                    break

            # we need to back up if we have blacklisted all clients
            predefined_max_len = int(self.cfg.oort_blacklist_max_len * len(self.totalArms))

            if len(blacklist) > predefined_max_len:
                logger.warning("Training Selector: exceeds the blacklist threshold")
                blacklist = blacklist[:predefined_max_len]

        return set(blacklist)
    
    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        aList.sort()
        clip_value = aList[min(int(len(aList) * clip_bound), len(aList)-1)]

        _max = max(aList)
        _min = min(aList)*0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList)/max(1e-4, float(len(aList)))

        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)
    
    def sample_clients(self, num_clients, available_clients, training_round, deadline):
        self.training_round = training_round
        self.blacklist = self.get_blacklist()
        scores = {}
        numOfExploited = 0
        exploreLen = 0

        client_list = list(self.totalArms.keys())
        orderedKeys = [x for x in client_list if x in available_clients and x not in self.blacklist]
        # print('len(client_list)', len(client_list), 'len(orderedKeys)',len(orderedKeys))

        # TODO: update the following code
        self.pacer()
        if self.round_threshold < 100.:
            sortedDuration = sorted([self.totalArms[key]['duration'] for key in client_list])
            self.round_prefer_duration = sortedDuration[min(int(len(sortedDuration) * self.round_threshold/100.), len(sortedDuration)-1)]
        else:
            self.round_prefer_duration = float('inf')
        # self.pacer()
        # self.round_prefer_duration = deadline


        moving_reward, staleness, allloss = [], [], {}  # staleness and allloss are never used

        for clientId in orderedKeys:
            # print(self.totalArms[clientId]['reward'])
            if self.totalArms[clientId]['reward'] > 0:
                creward = self.totalArms[clientId]['reward']
                moving_reward.append(creward)
                staleness.append(training_round - self.totalArms[clientId]['time_stamp'])

        # print('len(moving_reward)', len(moving_reward))
        max_reward, min_reward, range_reward, avg_reward, clip_value = self.get_norm(moving_reward, clip_bound=self.cfg.oort_clip_bound)
        # max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.get_norm(staleness, thres=1)  # never used

        for key in orderedKeys:
            # we have played this arm before
            if self.totalArms[key]['count'] > 0:
                creward = min(self.totalArms[key]['reward'], clip_value)
                numOfExploited += 1

                sc = (creward - min_reward)/float(range_reward) + math.sqrt(0.1 * math.log(training_round) / self.totalArms[key]['time_stamp'])

                clientDuration = self.totalArms[key]['duration']
                # print(clientDuration , self.round_prefer_duration)
                if clientDuration > self.round_prefer_duration:
                    sc *= (float(self.round_prefer_duration) / max(1e-4, clientDuration)) ** self.cfg.oort_round_penalty

                if self.totalArms[key]['time_stamp'] == training_round:
                    allloss[key] = sc

                scores[key] = sc  # utility in PPT


        clientLakes = list(scores.keys())
        self.exploration = max(self.exploration * self.decay_factor, self.exploration_min)
        exploitLen = min(int(num_clients * (1.0 - self.exploration)), len(clientLakes))

        # take the top-k, and then sample by probability, take 95% of the cut-off loss
        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)  # sort by value

        # take cut-off utility
        # print(scores, exploitLen, len(sortedClientUtil))
        cut_off_util = scores[sortedClientUtil[exploitLen]] * self.cfg.oort_cut_off_util

        pickedClients = []
        for clientId in sortedClientUtil:
            if scores[clientId] < cut_off_util:
                break
            pickedClients.append(clientId)
        # print(*[scores[i] for i in sortedClientUtil])
        # print(num_clients, self.exploration, exploitLen, cut_off_util)

        # augment_factor = len(pickedClients) # augment_factor is never used
        totalSc = max(1e-4, float(sum([scores[key] for key in pickedClients])))
        pickedClients = list(np.random.choice(pickedClients, exploitLen, p=[scores[key]/totalSc for key in pickedClients], replace=False))
        self.exploitClients = pickedClients
        # print('pickedClients', len(pickedClients), 'len(sortedClientUtil)', len(scores))

        # exploration: untrained clients
        if len(self.unexplored) > 0:
            _unexplored = [x for x in list(self.unexplored) if x in available_clients]

            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.totalArms[cl]['reward']
                clientDuration = self.totalArms[cl]['duration']

                if clientDuration > self.round_prefer_duration:
                    init_reward[cl] *= ((float(self.round_prefer_duration) / max(1e-4, clientDuration)) ** self.cfg.oort_round_penalty)

            # prioritize w/ some rewards (i.e., size)
            exploreLen = min(len(_unexplored), num_clients - len(pickedClients))
            pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window * exploreLen), len(init_reward))]

            unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))
            if unexploredSc >= 1e-4:
                pickedUnexplored = list(np.random.choice(pickedUnexploredClients, exploreLen,
                                p=[init_reward[key]/max(1e-4, unexploredSc) for key in pickedUnexploredClients], replace=False))
                
                # print('exploreLen', exploreLen, 'pickedUnexploredClients', len(pickedUnexploredClients), 'pickedUnexplored', len(pickedUnexplored))
                self.exploreClients = pickedUnexplored
                pickedClients = pickedClients + pickedUnexplored
        else:
            # no clients left for exploration
            self.exploration_min = 0.
            self.exploration = 0.

        # TODO: update the following code
        # while len(pickedClients) < num_clients:
        #     nextId = self.rdm.choice(orderedKeys)
        #     if nextId not in pickedClients:
        #         pickedClients.append(nextId)
        if len(pickedClients) < num_clients:
            remainedClients = set(orderedKeys) - set(pickedClients)
            pickedClients.extend(list(np.random.choice(list(remainedClients), num_clients - len(pickedClients), replace=False)))


        #TODO: the following code contributes NONE to the selection stage, comment it!
        # top_k_score = []
        # for i in range(min(3, len(pickedClients))):
        #     clientId = pickedClients[i]
        #     _score = (self.totalArms[clientId]['reward'] - min_reward)/range_reward
        #     _staleness = self.alpha*((training_round-self.totalArms[clientId]['time_stamp']) - min_staleness)/float(range_staleness) #math.sqrt(0.1*math.log(training_round)/max(1e-4, self.totalArms[clientId]['time_stamp']))
        #     top_k_score.append((self.totalArms[clientId], [_score, _staleness]))

        # logger.info("At round {}, UCB exploited {}, augment_factor {}, exploreLen {}, un-explored {}, exploration {}, round_threshold {}, sampled score is {}"
        #     .format(training_round, numOfExploited, augment_factor/max(1e-4, exploitLen), exploreLen, len(self.unexplored), self.exploration, self.round_threshold, top_k_score))
        # print('pickedClients', len(pickedClients), 'num_clients', num_clients)
        return pickedClients
