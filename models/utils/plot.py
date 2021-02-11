import json
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import pandas as pd
from collections import defaultdict


def plot_charge(file):
    img_dir = "../../data/img"
    fmt = '%Y-%m-%d %H:%M:%S'
    reference_time = "2020-01-02 00:00:00"
    color = ['black', 'blue', 'green', 'yellow', 'red']
    state = ['battery_charged_off', 'battery_charged_on', 'battery_low', 'battery_okay', 'phone_off', 'phone_on', 'screen_off', 'screen_on', 'screen_unlock']
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # key_list = list(data.keys())
    a = list(data.keys())
    random.shuffle(a)
    fig_num = 3
    fig_size = 50
    color = color * (fig_size // len(color) + 1)
    a = a[:fig_num * fig_size]
    for i in range(fig_num):
        plt.figure()
        for j in range(fig_size):
            start, end = None, None
            plot_j = False
            message = data[str(a[i * fig_num + j])]['messages'].split("\n")
            # for s in state:
                # message = message.replace(s, "\t" + s + "\n")
            # message = message.replace('\x00', '').strip().split("\n")
            for mes in message:
                try:
                    t, s = mes.strip().split("\t")
                    t = t.strip()
                    s = s.strip()
                    if s == 'battery_charged_on' and not start:
                        start = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(datetime.strptime(reference_time, fmt).timetuple())
                    elif s == 'battery_charged_off' and start:
                        end = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(datetime.strptime(reference_time, fmt).timetuple())
                        plt.plot([start, end], [j + 1, j + 1], color[j])
                        start, end = None, None
                        plot_j = True
                except:
                    pass
            if not plot_j:
                plt.plot([0], [j + 1], color[j])
        plt.xlabel("relative time")
        plt.ylabel("client id")
        plt.title("charged time")
        plt.savefig(os.path.join(img_dir, "relativeTime_clientID_{}.png".format(i + 1)), format="png")
    # plt.show()


def gen_json():
    data = pd.read_csv("../../data/user_behavior_tiny.csv", encoding="utf-8")
    data = data['extra']
    d = dict()
    for i in range(len(data)):
        d[i] = json.loads(data[i])
    with open("../../data/user_behavior_tiny.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4)


def static_ready(google=True):
    fmt = '%Y-%m-%d %H:%M:%S'
    reference_time = "2018-03-06 00:00:00"
    state = ['battery_charged_off', 'battery_charged_on', 'battery_low', 'battery_okay',
             'phone_off', 'phone_on', 'screen_off', 'screen_on', 'screen_unlock']
    with open("../../data/user_behavior_tiny.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    ready = defaultdict(list)
    for i in data:
        start_charge, end_charge, okay, low = None, None, None, None
        d = data[i]
        message = d['messages']
        uid = d['user_id']
        all = []
        for s in state:
            message = message.replace(s, "\t" + s + "\n")
        message = message.replace('\x00', '').strip().split("\n")
        for mes in message:
            t, s = mes.strip().split("\t")
            t = t.strip()
            s = s.strip()
            try:
                if s == 'battery_charged_on' and not start_charge:
                    start_charge = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(datetime.strptime(reference_time, fmt).timetuple())
                elif s == 'battery_charged_off' and start_charge:
                    end_charge = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(datetime.strptime(reference_time, fmt).timetuple())
                    all.append([start_charge, end_charge])
                    start_charge, end_charge = None, None

                if not google:
                    if s == 'battery_okay' and not okay:
                        okay = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(datetime.strptime(reference_time, fmt).timetuple())
                    elif s == 'battery_low' and okay:
                        low = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(datetime.strptime(reference_time, fmt).timetuple())
                        all.append([okay, low])
                        okay, low = None, None
            except:
                pass
        try:
            all = sorted(all, key=lambda x: x[0])
            now = all[0]
            for a in all:
                if now[1] >= a[0]:
                    now = [now[0], max(a[1], now[1])]
                else:
                    ready[uid].append(now)
                    now = a
            ready[uid].append(now)
        except:
            if len(ready[uid]) == 0:
                ready[uid] = []

    with open("../../data/ready_{}.json".format("strict" if google else "loose"), "w", encoding="utf-8") as f:
        json.dump(ready, f, indent=4)


def uid2behavior_tiny():
    with open('../../data/user_behavior_tiny.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    data = dict()
    for value in d.values():
        data[value['user_id']] = value
    with open('../../data/uid2behavior_tiny.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)


def plot_ready(extend=False, google=True):
    img_dir = "../../data/img"
    with open("../../data/ready{}_{}.json".format("_extend" if extend else "", "strict" if google else "loose"), "r", encoding="utf-8") as f:
        ready = json.load(f)
    client_cnt = []
    tmin, tmax = 0, 0
    static_section = 3 * 60
    min_section = 2 * 60
    for uid, ready_list in ready.items():
        for start, end in ready_list:
            if start > 0:
                tmin = min(tmin, start / static_section)
                tmax = max(tmax, end / static_section)
                client_cnt.extend(list(range(int(start / static_section), int((end + (static_section - min_section)) / static_section))))
            elif end > 0:
                tmax = max(tmax, end / static_section)
                client_cnt.extend(list(range(0, int((end + (static_section - min_section)) / static_section))))
    plt.hist(client_cnt, bins=int(tmax - tmin), color="blue")
    if not extend:
        plt.axis([0, int(4500 * 60 / static_section), 0, 1600])
    else:
        plt.axis([0, 7000, 0, 2000])
    plt.xlabel("relative time / {} min".format(int(static_section / 60)))
    plt.ylabel("client num")
    plt.title("relativeTime_clientNumber_{}.png".format("strict" if google else "loose"))
    plt.savefig(os.path.join(img_dir, "relativeTime_clientNumber{}_{}.png".
                             format("_extend" if extend else "", "strict" if google else "loose")), format="png")
    # plt.show()


def static_ready_extend(google=True):
    # if extend trace feature is okay
    fmt = '%Y-%m-%d %H:%M:%S'
    reference_time = "2018-03-06 00:00:00"
    refer_second = time.mktime(datetime.strptime(reference_time, fmt).timetuple())
    state = ['battery_charged_off', 'battery_charged_on', 'battery_low', 'battery_okay',
             'phone_off', 'phone_on', 'screen_off', 'screen_on', 'screen_unlock']
    with open("../../data/user_behavior_tiny.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("../../data/ready_{}.json".format("strict" if google else "loose"), "r", encoding="utf-8") as f:
        ready = json.load(f)
    ready_extend = defaultdict(list)

    for i in data:
        message = data[i]['messages']
        uid = data[i]['user_id']
        for s in state:
            message = message.replace(s, "\t" + s + "\n")
        message = message.replace('\x00', '').strip().split("\n")
        trace_start, trace_end = None, None
        for mes in message:
            t = mes.strip().split("\t")[0].strip()
            try:
                if not trace_start:
                    trace_start = time.mktime(datetime.strptime(t, fmt).timetuple()) - refer_second
                trace_end = time.mktime(datetime.strptime(t, fmt).timetuple()) - refer_second
            except:
                pass

        if not trace_start or not trace_end:
            continue
        T = trace_end - trace_start
        for p in range(10):
            ready_extend[uid].extend(list(map(lambda x: [x[0] + p * T, x[1] + p * T], ready[uid])))

    with open("../../data/ready_extend_{}.json".format("strict" if google else "loose"), "w", encoding="utf-8") as f:
        json.dump(ready_extend, f, indent=4)


if __name__ == '__main__':
    # static_ready_extend(True)
    # static_ready_extend(False)
    # plot_ready(True, True)
    # plot_ready(True, False)

    plot_charge('/home/ubuntu/storage/ycx/trace_sample/zipped_guid2data.json')
