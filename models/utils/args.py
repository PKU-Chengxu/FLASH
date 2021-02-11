import argparse

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']
SIM_TIMES = ['small', 'medium', 'large']


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config',
                    help='path to config file;',
                    type=str,
                    # required=True,
                    default='default.cfg')
    
    # move to config
    '''
    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    # required=True,
                    default='shakespeare')
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    # required=True,
                    default='stacked_lstm')
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    '''
    parser.add_argument('--metrics-name', 
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics-dir', 
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    # num epochs will be determined by config file
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    # move to config
    '''
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)
    parser.add_argument("-gpu_fraction", 
                        help="per process gpu memory fraction", 
                        type=float,
                        default=0.2,
                        required=False)
    '''

    return parser.parse_args()
