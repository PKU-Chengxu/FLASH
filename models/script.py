import os
import sys
if sys.argv[1] == '0':
    # os.system('python main.py --config femnist_gdrop_no_trace_5.cfg --metrics-name femnist_gdrop_no_trace_5')
    os.system('python main.py --config femnist_nocomp_no_trace_5.cfg --metrics-name femnist_nocomp_no_trace_5')
    os.system('python main.py --config femnist_sign_no_trace_5.cfg --metrics-name femnist_sign_no_trace_5')
else:
    # os.system('python main.py --config femnist_gdrop_trace_5.cfg --metrics-name femnist_gdrop_trace_5')
    os.system('python main.py --config femnist_nocomp_trace_5.cfg --metrics-name femnist_nocomp_trace_5')
    os.system('python main.py --config femnist_sign_trace_5.cfg --metrics-name femnist_sign_trace_5')