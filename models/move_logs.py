import os
import sys

dataset = sys.argv[1]
target_dir = '../exp_2_remake/fraction_abs/{}/'.format(dataset)
print('target: {}'.format(target_dir))

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

# os.system('mv clients_info_{}_* {}'.format(dataset, target_dir))
os.system('mv *_{}_*.json {}'.format(dataset, target_dir))
os.system('mv {}_*.cfg {}'.format(dataset, target_dir))
os.system('mv {}_*.log {}'.format(dataset, target_dir))
os.system('mv metrics/{}_*.csv {}'.format(dataset, target_dir))
# os.system('mv attended_clients_{}_* {}'.format(dataset, target_dir))
# os.system('mv metrics/{}_* {}'.format(dataset, target_dir))
