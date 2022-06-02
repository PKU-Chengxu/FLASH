#!/bin/bash

cfg=$1

if [[ $# -eq 0 ]]; then
    echo "use default config"
    cfg="default.cfg"
elif [[ $# -eq 1 ]]; then
    if [[ ! -f $cfg ]]; then
        echo "error: config file '${cfg}' does not exist, use defaul config"
        cfg="default.cfg"
    fi
else
    echo "error arg, please run: ./run.sh [CONFIG_FILE]"
    exit 1
fi




if [[ $cfg != "default.cfg" ]]; then
    cp ${cfg} ../models/
fi
cd ../models

python main.py --config `basename ${cfg}`