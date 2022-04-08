# FLASH

- An Open Source *Heterogeneity-Aware* Federated Learning Platform
- This repository is based on a fork of [Leaf](https://leaf.cmu.edu/), a benchmark for federated settings.

This repository contains the code and experiments for the paper:

>  [WWW'21](https://www2021.thewebconf.org/)
>
> [Characterizing Impacts of Heterogeneity in Federated Learning upon Large-Scale Smartphone Data]()

## What is FLASH?

Briefly speaking, we develop FLASH to incorporate **heterogeneity** into the federated learning simulation process. We mainly follow Google's [FL protocol](https://arxiv.org/pdf/1902.01046.pdf) to implement FLASH, so compared to other platforms, we add many additional system configurations, e.g., deadline. For these configurations, see more details in the [config file](#config).

### Heterogeneity

**Hardware Heterogeneity**: Each client is bundled with a device type. Each device type has different training speeds and network speeds. We also support self-defined device type(-1) whose parameter can be set manually for more complexed simulation. 

The source code for measure the on-device training time is available in the [OnDeviceTraining](./OnDeviceTraining) directory

**State(Behavior) Heterogeneity**: the state and running environment of participating clients can be various and dynamic. We follow [Google's FL system](https://arxiv.org/pdf/1902.01046.pdf), i.e., clients are available for training only when the device is idle, charging, and connected to WiFi. To simulate state heterogeneity, we provide a default state trace which can be accessed [here](./data/state_traces.json). This default trace is sampled from the large-scale real-world trace (as we use in our paper) that involves upto 136k devices.

Note: FLASH will run in a heterogeneity-unaware (ideal) mode if trace file is not found or `hard_hete` and `behav_hete` are set to `False`



## How to run it

### example

```bash
# 1. Clone and install requirments
git clone https://github.com/PKU-Chengxu/FLASH.git
pip3 install -r requirements.txt

# 2. Change state traces (optional)
# We have a provided a default state traces containing 1000 devices' data, located at the ./data/ dir. 
# IF you want to use a self-collected traces, just modify the file path in [models/client.py](models/client.py), i.e. with open('/path/to/state_traces.json', 'r', encoding='utf-8') as f: 

# 3. Download a benchmark dataset, go to directory of respective dataset `data/$DATASET` for instructions on generating the benchmark dataset

# 4. Run
cd models/
python3 main.py [--config yourconfig.cfg]
# use --config option to specify the config-file, default.cfg will be used if not specified
# the output log is CONFIG_FILENAME.log
```

<h3 id="config">Config File</h3>
To simplify the command line arguments, we move most of the parameters to a <span id="jump">config file</span>. Below is a detailed example.

```bash
## whether to consider heterogeneity
behav_hete False # bool, whether to simulate state(behavior) heterogeneity
hard_hete False # bool, whether to simulate hardware heterogeneity, which contains differential on-device training time and network speed


## no training mode to tune system configurations
no_training False # bool, whether to run in no_training mode, skip training process if True


## ML related configurations
dataset femnist # dataset to use
model cnn # file that defines the DNN model
learning_rate 0.01 # learning-rate of DNN
batch_size 10 # batch-size for training 


## system configurations, refer to https://arxiv.org/abs/1812.02903 for more details
num_rounds 500 # number of FL rounds to run
clients_per_round 100 # expected clients in each round
min_selected 60 # min selected clients number in each round, fail if not satisfied
max_sample 340 #  max number of samples to use in each selected client
eval_every 5 # evaluate every # rounds, -1 for not evaluate
num_epochs 5 # number of training epochs (E) for each client in each round
seed 0 # basic random seed
round_ddl 270 0 # μ and σ for deadline, which follows a normal distribution
update_frac 0.8  # min update fraction in each round, round fails when fraction of clients that successfully upload their is not less than "update_frac"
max_client_num -1 # max number of clients in the simulation process, -1 for infinite


### ----- NOTE! below are advanced configurations. 
### ----- Strongly recommend: specify these configurations only after reading the source code. 
### ----- Configuration items of [aggregate_algorithm, fedprox*, structure_k, qffl*] are mutually-exclusive 

## basic algorithm
aggregate_algorithm SucFedAvg # choose in [SucFedAvg, FedAvg], please refer to models/server.py for more details. In the configuration file, SucFedAvg refers to the "FedAvg" algorithm described in https://arxiv.org/pdf/1602.05629.pdf

## compression algorithm
# compress_algo grad_drop # gradiant compress algorithm, choose in [grad_drop, sign_sgd], not use if commented
# structure_k 100
## the k for structured update, not use if commented, please refer to the arxiv for more 

## advanced aggregation algorithms
# fedprox True # whether to apply fedprox and params needed, please refer to the sysml'20 (https://arxiv.org/pdf/1812.06127.pdf) for more details
# fedprox_mu 0.5
# fedprox_active_frac 0.8

# qffl True # whether to apply qffl(q-fedavg) and params needed, please refer to the ICLR'20 (https://arxiv.org/pdf/1905.10497.pdf) for more
# qffl_q 5
```


## Benchmark Datasets

#### FEMNIST

- **Overview:** Image Dataset
- **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
- **Task:** Image Classification



#### Celeba

- **Overview:** Image Dataset based on the [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Details:** 9343 users (we exclude celebrities with less than 5 images)
- **Task:** Image Classification (Smiling vs. Not smiling)



#### Reddit

- **Overview:** We preprocess the Reddit data released by [pushshift.io](https://files.pushshift.io/reddit/) corresponding to December 2017.
- **Details:** 1,660,820 users with a total of 56,587,343 comments. 
- **Task:** Next-word Prediction.



## Results in the paper

Config file and results are in the `paper_experiments` folder. You can just modify the `models/default.cfg` and then run `python main.py` to reproduce all the experiments in our paper. The experiments can be devided into the following categories:

- Basic FL algorithm
- Advanced FL algorithms
- Breakdown of Heterogeneity
- Device Failure
- Participation Bias



## On-device Training

the code we used to measure the on-device training time is in `OnDeviceTraining` folder. Please refer to the [doc](OnDeviceTraining/README.md) for more details



## Notes

- please consider to cite our paper if you use the code or data in your research project.


> ```
> @inproceedings{yang2019characterizing,
>   title={Characterizing impacts of heterogeneity in federated learning upon large-scale smartphone data},
>   author={Yang, Chengxu and Wang Qipeng and Xu, Mengwei and Chen, Zhenpeng and Bian Kaigui and Liu, Yunxin and Liu, Xuanzhe},
>   booktitle={The World Wide Web Conference},
>   year={2021}
> }
> ```
