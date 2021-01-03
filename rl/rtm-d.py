#* general
import time
import pandas as pd
from collections import namedtuple

from collections import defaultdict
from pymongo import MongoClient
from itertools import product, starmap

BENIGN_STATUS = 0
MALICIOUS_STATUS = 1
NUM_VEHICLES = 100
NUM_INTERACTIONS = 600
NUM_DATA_PER_CONTEXT = 500
DATA_PER_VEHICLE = 64 * NUM_DATA_PER_CONTEXT

result_values = defaultdict(lambda:[0,0,0,0]) #* precision, accuracy, recall
cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
trust_values = []
statuses = []
final = defaultdict(list)


def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']
    accrewcollection = db['rtm_d']
    return accrewcollection


def evaluate(threshold, data, interaction):
    decision = {}

    for i in range(NUM_VEHICLES):
        tv_id = trust_values[i]
        status = statuses[i]
        if tv_id > threshold:
            decision[i] = BENIGN_STATUS
        else:
            decision[i] = MALICIOUS_STATUS
        # print(interaction-i, decision[interaction-i], int(data['status'][interaction-i]))

        # * verify if its validity
        if decision[i] == MALICIOUS_STATUS and int(status) == MALICIOUS_STATUS:
            cases['gt'][0] += 1
        elif decision[i] == MALICIOUS_STATUS and int(status) == BENIGN_STATUS:
            cases['gt'][1] += 1
        elif decision[i] == BENIGN_STATUS and int(status) == MALICIOUS_STATUS:
            cases['gt'][2] += 1
        else:
            cases['gt'][3] += 1
    # print(cases)
    #* calucalte precision, accuracy, recall
    #* precision
    if (cases["gt"][0] + cases["gt"][1]) == 0:
        result_values[interaction][0]=0
    else:
        result_values[interaction][0] = (cases["gt"][0])/(cases["gt"][0] + cases["gt"][1]) *100
    #* accuracy
    if (cases["gt"][0] + cases["gt"][1] +  cases["gt"][2] + cases["gt"][3]) ==0:
        result_values[interaction][1] =0
    else:
        result_values[interaction][1] = (cases["gt"][0] + cases["gt"][3])/(cases["gt"][0] + cases["gt"][1] + cases["gt"][2] + cases["gt"][3]) *100
    #* recall
    if (cases["gt"][0] + cases["gt"][2]) == 0:
        result_values[interaction][2]=0
    else:
        result_values[interaction][2] = (cases["gt"][0])/(cases["gt"][0] + cases["gt"][2]) *100
    #* f1 
    if (result_values[interaction][0]+result_values[interaction][2]) ==0:
        result_values[interaction][3]=0
    else:
        result_values[interaction][3]=(2*result_values[interaction][2]*result_values[interaction][0]) / (result_values[interaction][0] + result_values[interaction][2])
    # print(interaction, result_values[interaction])
    # print(interaction, cases)
    final['acc'].append(result_values[interaction][0])
    final['pre'].append(result_values[interaction][1])
    final['rec'].append(result_values[interaction][2])
    final['f1'].append(result_values[interaction][2])

    # return result_values

def get_indirect_trust_values(data):
    for i in range(NUM_VEHICLES):
        # print(interaction-i)
        # *** Not too clear about how mongoDB's index is working - Might need to -1 from the index. ***
        index = DATA_PER_VEHICLE * i + NUM_DATA_PER_CONTEXT
        good_history = data['good_history'][index]
        bad_history = data['bad_history'][index]
        trust_values.append(float(good_history / (bad_history + good_history) * 100))
        statuses.append(data['status'][index])


connection = connect()
for output in named_product(v_s = [59999], v_mvp=[0.2], v_mbp=[0.5], v_oap=[0.2], interval=[100]):
    threshold=50
    PPV_THR = 0.8
    NPV_THR = 0.8

    filename = "ce_db_0_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
    data = pd.read_csv('../sampledata/'+filename, sep=',', error_bad_lines=False, encoding='latin1', header=0)
    interaction=1
    get_indirect_trust_values(data)
    while True:
        if interaction == NUM_INTERACTIONS:
            row = {"id": str(output), 'v_mvp': output.v_mvp, 'v_mbp': output.v_mbp, 'v_oap': output.v_oap,
                   "accuracy": final['acc'], 'precision': final['pre'], 'recall': final['rec']}
            # connection.insert_one(row)
            # print(row)
            break
        evaluate(threshold, data, interaction)

        PPV = cases['gt'][0] / (cases['gt'][0] + cases['gt'][1])
        NPV = cases['gt'][3] / (cases['gt'][3] + cases['gt'][2])

        # if PPV > PPV_THR and NPV > NPV_THR:
        #     threshold += 5
        #     if threshold > 100:
        #         threshold = 100
        # elif NPV<NPV_THR:
        #     threshold -= 5
        #     if threshold < 0:
        #         threshold = 0
        # elif PPV < PPV_THR:
        #     threshold -= 5
        #     if threshold < 0:
        #         threshold = 0
        

        if NPV < NPV_THR:
            threshold += 5
            if threshold > 100:
                threshold = 100
        elif PPV < PPV_THR:
            threshold -= 5
            if threshold < 0:
                threshold = 0
        print(interaction, threshold, PPV, NPV)
        interaction += 1
        print(cases)

    cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
    result_values = defaultdict(lambda:[0,0,0,0]) #* precision, accuracy, recall,f1

    final = defaultdict(list)
