#* general
import time
import pandas as pd
from collections import namedtuple

from collections import defaultdict
from pymongo import MongoClient
from itertools import product, starmap

result_values = defaultdict(lambda:[0,0,0]) #* precision, accuracy, recall
cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
final = defaultdict(list)
def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']
    accrewcollection = db['rtm-d']
    return accrewcollection

def evaluate(threshold, interval, data, nci):
    decision = {}
    
    for i in range(interval):
        # print(nci-i)

        #* make decision
        tv_id = int(data['indirect_tv'][nci-i]*100)
        if tv_id > threshold:
            decision[nci-i]=0
        else:
            decision[nci-i]=1
        # print(nci-i, decision[nci-i], int(data['status'][nci-i]))

        #* verify if its validity
        if decision[nci-i] == 1 and int(data['status'][nci-i])==1:
            cases['gt'][0]+=1
        elif decision[nci-i] == 1 and int(data['status'][nci-i])==0:
            cases['gt'][1]+=1
        elif decision[nci-i] == 0 and int(data['status'][nci-i])==1:
            cases['gt'][2]+=1
        else:
            cases['gt'][3]+=1
    # print(cases)
    #* calucalte precision, accuracy, recall
    #* precision
    if (cases["gt"][0] + cases["gt"][1]) == 0:
        result_values[nci][0]=100
    else:
        result_values[nci][0] = (cases["gt"][0])/(cases["gt"][0] + cases["gt"][1]) *100 
    #* accuracy
    if (cases["gt"][0] + cases["gt"][1] +  cases["gt"][2] + cases["gt"][3]) ==0:
        result_values[nci][1] =100
    else:
        result_values[nci][1] = (cases["gt"][0] + cases["gt"][3])/(cases["gt"][0] + cases["gt"][1] + cases["gt"][2] + cases["gt"][3]) *100 
    #* recall
    if (cases["gt"][0] + cases["gt"][2]) == 0:
        result_values[nci][2]=100
    else:
        result_values[nci][2] = (cases["gt"][0])/(cases["gt"][0] + cases["gt"][2]) *100
    # print(nci, result_values[nci]) 
    # print(nci, cases)
    final['acc'].append(result_values[nci][0])
    final['pre'].append(result_values[nci][1])
    final['rec'].append(result_values[nci][2])
    # return result_values

connection = connect()
for output in named_product(v_s = [11000], v_mvp=[0.2], v_mbp=[0.5], v_oap=[0.2, 0.4], v_interval=[100]): 
    threshold=50

    filename = "rl_df_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
    data = pd.read_csv('../sampledata/'+filename, header=0)
    INTERVAL = output.v_interval
    next_car_index=1
    time.sleep(0.1)
    while True:

        if next_car_index % INTERVAL ==0:
            evaluate(threshold, INTERVAL, data, next_car_index)
        # if next_car_index % 200 == 0:

            PPV = cases['gt'][0] / (cases['gt'][0] + cases['gt'][1])
            if PPV < 0.95:
                threshold-=5
                if threshold <0:
                    threshold=0
            NPV = cases['gt'][3] / (cases['gt'][3] + cases['gt'][2])
            if NPV < 0.95:
                threshold+=5
                if threshold>100:
                    threshold=100
            print(next_car_index,  threshold, PPV, NPV)

        if next_car_index == (output.v_s):
            row = {"id": str(output), 'v_mvp': output.v_mvp, 'v_mbp': output.v_mbp, 'v_oap': output.v_oap, 'v_interval':output.v_interval, "v_s": output.v_s, "accuracy": final['acc'], 'precision': final['pre'], 'recall': final['rec']}
            # connection.insert_one(row)
            print(final['acc'][-1])
            break

        next_car_index+=1
    cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
    result_values = defaultdict(lambda:[0,0,0]) #* precision, accuracy, recall

    final = defaultdict(list)
