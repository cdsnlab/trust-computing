#! done here!
import pandas as pd
from collections import namedtuple
import random
import statistics
from collections import defaultdict
from pymongo import MongoClient
from itertools import product, starmap

result_values = defaultdict(lambda:[0,0,0, 0]) #* precision, accuracy, recall, f1
cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN

cum_accuracy = defaultdict(list)
step_accuracy=defaultdict(list)
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
    accrewcollection = db['dtm_pnt']
    return accrewcollection

def evaluate(threshold, samplelist, data, intnumb):
    decision = {}
    cases['recent'] = [0,0,0,0]

    for i in (samplelist):
        # print(nci-i)

        #* make decision
        tv_d = int(data['direct_tv'][i]*100)
        # print(nci-i, tv_d)
        if tv_d > threshold:
            decision[i]=0
        else:
            decision[i]=1
        # print(nci-i, decision[nci-i], int(data['status'][nci-i]))

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

    #* calucalte precision, accuracy, recall
    #* precision
    if (cases["gt"][0] + cases["gt"][1]) == 0:
        result_values[intnumb][0]=0
    else:
        result_values[intnumb][0] = (cases["gt"][0])/(cases["gt"][0] + cases["gt"][1]) *100 
    #* accuracy
    if (cases["gt"][0] + cases["gt"][1] +  cases["gt"][2] + cases["gt"][3]) ==0:
        result_values[intnumb][1] =0
    else:
        result_values[intnumb][1] = (cases["gt"][0] + cases["gt"][3])/(cases["gt"][0] + cases["gt"][1] + cases["gt"][2] + cases["gt"][3]) *100 
    #* recall
    if (cases["gt"][0] + cases["gt"][2]) == 0:
        result_values[intnumb][2]=0
    else:
        result_values[intnumb][2] = (cases["gt"][0])/(cases["gt"][0] + cases["gt"][2]) *100
    #* f1 score
    if result_values[intnumb][2]+result_values[intnumb][0] == 0:
        result_values[intnumb][3]=0
    else:
        result_values[intnumb][3]= (2*result_values[intnumb][2]*result_values[intnumb][0]) / (result_values[intnumb][2] + result_values[intnumb][0])


def step_records(run_counts, intnumb, threshold):
    cum_accuracy[run_counts].append(result_values[intnumb][1])
    precision[run_counts].append(result_values[intnumb][0])
    recall[run_counts].append(result_values[intnumb][2])
    f1score[run_counts].append(result_values[intnumb][3])
    average_dtt[run_counts].append(threshold)
    step_accuracy[run_counts].append((cases["recent"][0] + cases["recent"][3])/(cases["recent"][0] + cases["recent"][1] + cases["recent"][2] + cases["recent"][3]) *100)
    # print(step_accuracy)


def get_sample(max_value):
    sample_index = random.sample(range(max_value), 100)
    return sample_index

def save_avg_accuracy(run_counts, name, steps):
    final_cum_acc, final_dtt, final_gt, final_rew, final_precision, final_recall, final_acc_error, final_f1, final_avg_acc= [], [], [], [], [], [], [], [], []
    for j in range(steps):
        
        temp ={0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0} #acc, dtt, gt, rew
        errors=[]
        for i in range(run_counts):
            temp[0]+=cum_accuracy[i][j]  
            temp[1]+=average_dtt[i][j] 
            temp[4]+=precision[i][j] 
            temp[5]+=recall[i][j] 
            temp[6]+=f1score[i][j]
            temp[8]+=step_accuracy[i][j]
            errors.append(cum_accuracy[i][j])    

        final_acc_error.append(statistics.stdev(errors))
        final_cum_acc.append(temp[0]/run_counts)
        final_dtt.append(temp[1]/run_counts)
        final_precision.append(temp[4]/run_counts)
        final_recall.append(temp[5]/run_counts)
        final_f1.append(temp[6]/run_counts)
        final_avg_acc.append(temp[8]/run_counts)

    print("Accuracy: ", final_cum_acc[-1])
    print("Precision: ", final_precision[-1])
    print("Recall: ", final_recall[-1])
    print("F1 score: ", final_f1[-1])
    row = {"id": str(output), 'v_i': output.v_i, 'v_mvp': output.v_mvp, 'v_mbp': output.v_mbp, 'v_oap': output.v_oap, "v_s": output.v_s, "cum_accuracy": final_cum_acc, "step_accuracy": final_avg_acc,'precision': final_precision, 'recall': final_recall, 'dtt': final_dtt, 'f1score': final_f1}
    connection.insert_one(row)

connection = connect()
for output in named_product(v_i = [10, 50, 90], v_s = [12000], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.1, 0.3, 0.5, 0.7, 0.9], v_oap=[0.1, 0.15, 0.2, 0.25, 0.3], v_ppvnpvthr=[0.1, 0.5, 0.9]): 
# for output in named_product(v_i = [10], v_s = [12000], v_mvp=[0.1], v_mbp=[0.1], v_oap=[0.1], v_ppvnpvthr=[0.1, 0.5, 0.9]): 

    threshold=output.v_i

    filename = "cares_df_0_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
    data = pd.read_csv('../../sampledata/'+filename, header=0)
    STEPS = output.v_s

    run_counts = 20

    for i in range (run_counts):
        interaction_number=1
        while True:
            samplelist = get_sample(STEPS)
            evaluate(threshold, samplelist, data, interaction_number)
            step_records(i, interaction_number, threshold)
            # print(cases)
            if interaction_number == (STEPS)/100:
                print("run count: {} finished ".format(i))
                print("dtt {}".format(threshold))
                print("Accuracy {}".format(cum_accuracy[i][-1]))
                threshold=output.v_i
        
                break
            interaction_number+=1
        cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
        result_values = defaultdict(lambda:[0,0,0,0]) #* precision, accuracy, recall

    save_avg_accuracy(run_counts, output, int((STEPS+1)/100))
    cum_accuracy = defaultdict(list)
    step_accuracy = defaultdict(list)
    precision = defaultdict(list)
    recall = defaultdict(list)
    f1score = defaultdict(list)
    average_dtt = defaultdict(list)
