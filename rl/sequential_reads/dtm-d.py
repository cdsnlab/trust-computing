#* general
import time
import pandas as pd
from collections import namedtuple

from collections import defaultdict
from pymongo import MongoClient
from itertools import product, starmap

result_values = defaultdict(lambda:[0,0,0, 0]) #* precision, accuracy, recall, f1
cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
final = defaultdict(list)
def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']
    accrewcollection = db['dtm_d95']
    return accrewcollection

def evaluate(threshold, interval, data, nci):
    decision = {}
    cases['recent'] = [0,0,0,0]

    for i in range(interval):
        # print(nci-i)

        #* make decision
        tv_d = int(data['direct_tv'][nci-i]*100)
        # print(nci-i, tv_d)
        if tv_d > threshold:
            decision[nci-i]=0
        else:
            decision[nci-i]=1
        # print(nci-i, decision[nci-i], int(data['status'][nci-i]))

        #* verify if its validity
        if decision[nci-i] == 1 and int(data['status'][nci-i]==1): #* TP: malicious!
            cases['gt'][0]+=1
            cases['recent'][0]+=1

        elif decision[nci-i] == 1 and int(data['status'][nci-i]==0): #* FP: thought malicious but not!
            cases['gt'][1]+=1
            cases['recent'][1]+=1
        elif decision[nci-i] == 0 and int(data['status'][nci-i]==1):
            cases['gt'][2]+=1
            cases['recent'][2]+=1
        else:
            cases['gt'][3]+=1
            cases['recent'][3]+=1



    
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
    #* f1 score
    if result_values[nci][2]+result_values[nci][0] == 0:
        result_values[nci][3]=0
    else:
        result_values[nci][3]= (2*result_values[nci][2]*result_values[nci][0]) / (result_values[nci][2] + result_values[nci][0]) 
    # print(nci, result_values[nci], threshold) 
    # print(nci, result_values[nci]) 
    # print(nci, cases)
    final['acc'].append(result_values[nci][1])
    final['pre'].append(result_values[nci][0])
    final['rec'].append(result_values[nci][2])
    final['f1'].append(result_values[nci][3])
    final['dtt'].append(threshold)


connection = connect()
for output in named_product(v_i = [10, 50, 90], v_s = [59999], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.1, 0.2, 0.3, 0.4, 0.5], v_oap=[0.1, 0.15, 0.2, 0.25, 0.3]): 
# for output in named_product(v_i = [10, 50, 90], v_s = [59999], v_mvp=[0.2], v_mbp=[0.5], v_oap=[0.2]): 

    threshold=output.v_i

    PPV_THR = 0.95
    NPV_THR = 0.95
    hitmax=False
    hitmin=False

    filename = "cares_df_0_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
    data = pd.read_csv('../sampledata/'+filename, header=0)
    INTERVAL = 100
    next_car_index=1
    # time.sleep(0.1)
    while True:

        if next_car_index % INTERVAL ==0:
            evaluate(threshold, INTERVAL, data, next_car_index)

            if (cases['recent'][0] + cases['recent'][1]) ==0:
                PPV=0
            else:
                PPV = cases['recent'][0] / (cases['recent'][0] + cases['recent'][1])
            if (cases['recent'][3] + cases['recent'][2])==0:
                NPV =0
            else:
                NPV = cases['recent'][3] / (cases['recent'][3] + cases['recent'][2])

            #! 방법1: 둘다  상쇄되는 문제가 발생!
            # if PPV < 0.95:
            #     threshold-=5
            #     if threshold<0:
            #         threshold=0
            # if NPV > 0.95:
            #     threshold+=5
            #     if threshold >100:
            #         threshold=100

            #! 방법2: 신기한 PPV, NPV기반 방법 -> 
            # if (PPV > PPV_THR and NPV > NPV_THR): #* 둘다 1에 가까우면 움직이지마!
            #     pass
            # elif(NPV < NPV_THR): #* 둘중 하나를 고쳐야되면 NPV부터 고쳐봐. 
            #     if threshold + 5 < 100:
            #         threshold+=5
            # elif(PPV < PPV_THR):
            #     if threshold - 5 > 0:
            #         threshold-=5

            #! 방법3: 완전히 올라가버리는 현상 제거 (꼼수?)
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

            #! 방법4
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
        
            # print(next_car_index,  threshold, PPV, NPV, final['acc'][-1])

        if next_car_index == (output.v_s):
            row = {"id": str(output), 'v_i': output.v_i, 'v_mvp': output.v_mvp, 'v_mbp': output.v_mbp, 'v_oap': output.v_oap, "v_s": output.v_s, "accuracy": final['acc'], 'precision': final['pre'], 'recall': final['rec'], 'dtt': final['dtt'], 'f1score': final['f1']}
            connection.insert_one(row)
            print(final['acc'][-1])
            break

        next_car_index+=1
    cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
    result_values = defaultdict(lambda:[0,0,0,0]) #* precision, accuracy, recall

    final = defaultdict(list)