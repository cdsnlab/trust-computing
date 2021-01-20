#* general
import time
import pandas as pd
from collections import namedtuple
import random
import statistics

from collections import defaultdict
from pymongo import MongoClient
from itertools import product, starmap

result_values = defaultdict(lambda:[0,0,0, 0]) #* precision, accuracy, recall, f1
cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN

accuracy = defaultdict(list)
precision = defaultdict(list)
recall = defaultdict(list)
f1score = defaultdict(list)
average_dtt= defaultdict(list)

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']
    accrewcollection = db['istm_d95_r']
    return accrewcollection

def evaluate(threshold, samplelist, data, intnumb, beta):
    decision = {}
    cases['recent'] = [0,0,0,0]

    for i in (samplelist):

        #* make decision
        tv_d = int(data['direct_tv'][i]*100)
        tv_id = int(data['indirect_tv'][i]*100)
        tv = (beta * tv_id + (1-beta) * tv_d)
        if tv > threshold:
            decision[i]=0
        else:
            decision[i]=1
        #* verify if its validity
        if decision[i] == 1 and int(data['status'][i]==1): #* TP: malicious!
            cases['gt'][0]+=1
            cases['recent'][0]+=1

        elif decision[i] == 1 and int(data['status'][i]==0): #* FP: thought malicious but not!
            cases['gt'][1]+=1
            cases['recent'][1]+=1
        elif decision[i] == 0 and int(data['status'][i]==1):
            cases['gt'][2]+=1
            cases['recent'][2]+=1
        else:
            cases['gt'][3]+=1
            cases['recent'][3]+=1
    # print(cases)
    #* calucalte precision, accuracy, recall    
    #* precision
    if (cases["gt"][0] + cases["gt"][1]) == 0:
        result_values[intnumb][0]=100
    else:
        result_values[intnumb][0] = (cases["gt"][0])/(cases["gt"][0] + cases["gt"][1]) *100 
    #* accuracy
    if (cases["gt"][0] + cases["gt"][1] +  cases["gt"][2] + cases["gt"][3]) ==0:
        result_values[intnumb][1] =100
    else:
        result_values[intnumb][1] = (cases["gt"][0] + cases["gt"][3])/(cases["gt"][0] + cases["gt"][1] + cases["gt"][2] + cases["gt"][3]) *100 
    #* recall
    if (cases["gt"][0] + cases["gt"][2]) == 0:
        result_values[intnumb][2]=100
    else:
        result_values[intnumb][2] = (cases["gt"][0])/(cases["gt"][0] + cases["gt"][2]) *100
    #* f1 score
    if result_values[intnumb][2]+result_values[intnumb][0] == 0:
        result_values[intnumb][3]=0
    else:
        result_values[intnumb][3]= (2*result_values[intnumb][2]*result_values[intnumb][0]) / (result_values[intnumb][2] + result_values[intnumb][0]) 


def step_records(run_counts, intnumb, threshold):
    accuracy[run_counts].append(result_values[intnumb][1])
    precision[run_counts].append(result_values[intnumb][0])
    recall[run_counts].append(result_values[intnumb][2])
    f1score[run_counts].append(result_values[intnumb][3])
    average_dtt[run_counts].append(threshold)


def get_sample(max_value):
    sample_index = random.sample(range(max_value), 100)
    return sample_index

def save_avg_accuracy(run_counts, name, steps):
    final_acc, final_dtt, final_gt, final_rew, final_precision, final_recall, final_acc_error, final_f1= [], [], [], [], [], [], [], []
    for j in range(steps):
        
        temp ={0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0} #acc, dtt, gt, rew
        errors=[]
        for i in range(run_counts):
            temp[0]+=accuracy[i][j]  
            temp[1]+=average_dtt[i][j] 
            temp[4]+=precision[i][j] 
            temp[5]+=recall[i][j] 
            temp[6]+=f1score[i][j]
            errors.append(accuracy[i][j])    

        final_acc_error.append(statistics.stdev(errors))
        final_acc.append(temp[0]/run_counts)
        final_dtt.append(temp[1]/run_counts)
        final_precision.append(temp[4]/run_counts)
        final_recall.append(temp[5]/run_counts)
        final_f1.append(temp[6]/run_counts)
    print("Accuracy: ", final_acc[-1])
    print("Precision: ", final_precision[-1])
    print("Recall: ", final_recall[-1])
    print("F1 score: ", final_f1[-1])
    row = {"id": str(output), 'v_i': output.v_i, 'v_mvp': output.v_mvp, 'v_mbp': output.v_mbp, 'v_oap': output.v_oap, "v_s": output.v_s, "accuracy": final_acc, 'precision': final_precision, 'recall': final_recall, 'dtt': final_dtt, 'f1score': final_f1}
    connection.insert_one(row)

connection = connect()
for output in named_product(v_i = [10, 50, 90], v_s = [59999], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.1, 0.2, 0.3, 0.4, 0.5], v_oap=[0.1, 0.15, 0.2, 0.25, 0.3]): 
# for output in named_product(v_i = [10, 50, 90], v_s = [59999], v_mvp=[0.2], v_mbp=[0.5], v_oap=[0.2]): 
    threshold=output.v_i
    STEPS = output.v_s
    PPV_THR = 0.95
    NPV_THR = 0.95
    hitmax=False
    hitmin=False

    filename = "is_df_0_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
    data = pd.read_csv('../../sampledata/'+filename, header=0)
    BETA = 0.5
    run_counts = 10
    for i in range (run_counts):
        interaction_number=1
        while True:
            samplelist = get_sample(STEPS)
            evaluate(threshold, samplelist, data, interaction_number, BETA)
            step_records(i, interaction_number, threshold)
        
            if (cases['recent'][0] + cases['recent'][1]) ==0:
                PPV=0
            else:
                PPV = cases['recent'][0] / (cases['recent'][0] + cases['recent'][1])
            if (cases['recent'][3] + cases['recent'][2])==0:
                NPV =0
            else:
                NPV = cases['recent'][3] / (cases['recent'][3] + cases['recent'][2])

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

            if interaction_number == (STEPS+1)/100:
                print("run count: {} finished ".format(i))
                print("dtt {}".format(threshold))
                print("Accuracy {}".format(accuracy[i][-1]))
                threshold=output.v_i

                break
            interaction_number+=1
        cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
        result_values = defaultdict(lambda:[0,0,0,0]) #* precision, accuracy, recall

    save_avg_accuracy(run_counts, output, int((STEPS+1)/100))
    accuracy = defaultdict(list)
    precision = defaultdict(list)
    recall = defaultdict(list)
    f1score = defaultdict(list)        
    average_dtt= defaultdict(list)
