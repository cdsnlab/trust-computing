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
    accrewcollection = db['rtm_d95']
    return accrewcollection


def evaluate(threshold, data, interaction):
    decision = {}
    # print(trust_values)
    cases['recent'] = [0,0,0,0]
    for i in range(NUM_VEHICLES):
        tv_id = trust_values[i]
        status = statuses[i]
        # print(tv_id, threshold)
        if tv_id > threshold:
            decision[i] = BENIGN_STATUS
        else:
            decision[i] = MALICIOUS_STATUS
        # print(interaction-i, decision[interaction-i], int(data['status'][interaction-i]))

        # * verify if its validity
        if decision[i] == MALICIOUS_STATUS and int(status) == MALICIOUS_STATUS:
            cases['gt'][0] += 1
            cases['recent'][0]+=1
        elif decision[i] == MALICIOUS_STATUS and int(status) == BENIGN_STATUS:
            cases['gt'][1] += 1
            cases['recent'][1]+=1
        elif decision[i] == BENIGN_STATUS and int(status) == MALICIOUS_STATUS:
            cases['gt'][2] += 1
            cases['recent'][2]+=1
        else:
            cases['gt'][3] += 1
            cases['recent'][3]+=1
    # print(cases['recent'])
    # print(decision, statuses)
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
        result_values[interaction][1] = float((cases["gt"][0] + cases["gt"][3])/(cases["gt"][0] + cases["gt"][1] + cases["gt"][2] + cases["gt"][3]) *100)
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
    final['acc'].append(result_values[interaction][1])
    final['pre'].append(result_values[interaction][0])
    final['rec'].append(result_values[interaction][2])
    final['f1'].append(result_values[interaction][3])
    final['dtt'].append(threshold)
    # return result_values

def get_indirect_trust_values(data):
    for i in range(NUM_VEHICLES):
        # print(interaction-i)
        # *** Not too clear about how mongoDB's index is working - Might need to -1 from the index. ***
        index = DATA_PER_VEHICLE * i + NUM_DATA_PER_CONTEXT
        good_history = data['good_history'][index-1] #* 100 instance
        bad_history = data['bad_history'][index-1] #*100 instance
        trust_values.append(float(good_history / (bad_history + good_history) * 100))
        statuses.append(data['status'][index-1])
        # print(i, good_history, bad_history)


connection = connect()
for output in named_product(v_i=[10, 50, 90],v_s = [59999], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.1, 0.2, 0.3, 0.4, 0.5], v_oap=[0.1, 0.15, 0.2, 0.25, 0.3]): 
# for output in named_product(v_i=[10,50,90],v_s = [59999], v_mvp=[0.1], v_mbp=[0.1], v_oap=[0.1]):
    threshold=output.v_i
#
    PPV_THR = 0.95
    NPV_THR = 0.95
    hitmax=False
    hitmin=False
    filename = "ce_db_0_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
    data = pd.read_csv('../sampledata/'+filename, sep=',', error_bad_lines=False, encoding='latin1', header=0, nrows=3200000)
    interaction=1
    get_indirect_trust_values(data)
    # print(trust_values)
    while True:
        if interaction == NUM_INTERACTIONS:
            row = {"id": str(output), 'v_i': output.v_i, 'v_mvp': output.v_mvp, 'v_mbp': output.v_mbp, 'v_oap': output.v_oap, "v_s": output.v_s, "accuracy": final['acc'], 'precision': final['pre'], 'recall': final['rec'], 'dtt': final['dtt'], 'f1score': final['f1']}
            connection.insert_one(row)
            print("{} finished ".format(output))

            # print(row)
            break
        evaluate(threshold, data, interaction)
        if (cases['recent'][0] + cases['recent'][1]) ==0:
            PPV=0
        else:
            PPV = cases['recent'][0] / (cases['recent'][0] + cases['recent'][1])
        if (cases['recent'][3] + cases['recent'][2])==0:
            NPV =0
        else:
            NPV = cases['recent'][3] / (cases['recent'][3] + cases['recent'][2])
        #* 방법3
        # if (PPV > PPV_THR and NPV > NPV_THR): #* 둘다 1에 가까우면 움직이지마!
        #     pass
        # # elif(PPV < PPV_THR and NPV < NPV_THR ):
        # #     if threshold > 50:
        # #         threshold-=5
        # #     else:
        # #         threshold+=5
        # elif(NPV < NPV_THR): #* 둘중 하나를 고쳐야되면 NPV부터 고쳐봐. 
        #     if threshold + 5 < 100:
        #         threshold+=5
                
        # elif(PPV < PPV_THR):
        #     if threshold - 5 > 0:
        #         threshold-=5
        #* 방법 4: max찍으면 내려가삼.
        
        if (PPV > PPV_THR and NPV > NPV_THR): #* 둘다 1에 가까우면 움직이지마!
            pass 
        elif hitmax or hitmin: #* max, min 찍으면 반대방향으로 이동해.
            if hitmax:
                if threshold - 5 > 0:
                    threshold-=5
            if hitmin:
                if threshold + 5 < 100:
                    threshold+=5
        
        elif(NPV < NPV_THR): #* 둘중 하나를 고쳐야되면 NPV부터 고쳐봐. 
            if threshold + 5 < 100:
                threshold+=5
                
        elif(PPV < PPV_THR):
            if threshold - 5 > 0:
                threshold-=5

        if threshold == 95:
            hitmax=True
            hitmin=False
        if threshold == 5:
            hitmin=True
            hitmax=False

        print(interaction, threshold, PPV, NPV, final['acc'][-1])
        interaction += 1
        # print(cases)

    cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
    result_values = defaultdict(lambda:[0,0,0,0]) #* precision, accuracy, recall,f1
    trust_values = []
    statuses = []
    final = defaultdict(list)

