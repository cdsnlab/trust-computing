#* general
import time
import pandas as pd
from collections import namedtuple
import random
import statistics
from collections import defaultdict
from pymongo import MongoClient
from itertools import product, starmap
from slack_noti import slacknoti

result_values = defaultdict(lambda:[0,0,0,0]) #* precision, accuracy, recall, f1
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
    accrewcollection = db['rtm_d_manual']
    return accrewcollection


def evaluate(threshold, samplelist, data, intnumb):
    decision = {}
    cases['recent'] = [0,0,0,0]

    for i in (samplelist):
        good_history = data['good_history'][i]
        bad_history = data['bad_history'][i]
        tv_id = int(good_history / (bad_history + good_history) * 100)

        if tv_id >= threshold: #! added equ
            decision[i]=0
        else:
            decision[i]=1
        # print(good_history, bad_history, tv_id)
        # print(interaction-i, decision[interaction-i], int(data['status'][interaction-i]))

        #* verify if its validity
        if decision[i] == 1 and int(data['status'][i]==1): #* TP: malicious!
            cases['gt'][0]+=1
            cases['recent'][0]+=1
        elif decision[i] == 1 and int(data['status'][i]==0): #* FP: thought malicious but not!
            cases['gt'][1]+=1
            cases['recent'][1]+=1
        elif decision[i] == 0 and int(data['status'][i]==1): #* FN
            cases['gt'][2]+=1
            cases['recent'][2]+=1
        else: #* TN
            cases['gt'][3]+=1
            cases['recent'][3]+=1
    # print(cases['recent'])
        # print(i, decision[i], data['status'][i])
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

def get_available_list(data):
    availablelist=[]
    for i in range(100*64):
    # for i in range(100):
        # print(interaction-i)
        # index = DATA_PER_VEHICLE * i + NUM_DATA_PER_CONTEXT
        # index = 500 * i + 500
        index = 500 * i + 499
        good_history = data['good_history'][index] #* 100 instance
        bad_history = data['bad_history'][index] #*100 instance
        # print(good_history, bad_history)
        availablelist.append(index)
    # print(availablelist)
    return availablelist
    # return random.sample(availablelist,100)

def step_records(run_counts, intnumb, threshold):
    cum_accuracy[run_counts].append(result_values[intnumb][1])
    precision[run_counts].append(result_values[intnumb][0])
    recall[run_counts].append(result_values[intnumb][2])
    f1score[run_counts].append(result_values[intnumb][3])
    average_dtt[run_counts].append(threshold)
    step_accuracy[run_counts].append((cases["recent"][0] + cases["recent"][3])/(cases["recent"][0] + cases["recent"][1] + cases["recent"][2] + cases["recent"][3]) *100)

    # print(accuracy)

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
    row = {"id": str(output), 'v_i': output.v_i, 'v_mvp': output.v_mvp, 'v_mbp': output.v_mbp, 'v_oap': output.v_oap, "v_s": output.v_s, "cum_accuracy": final_cum_acc, "step_accuracy": final_avg_acc,'precision': final_precision, 'recall': final_recall, 'dtt': final_dtt, 'f1score': final_f1, 'v_ppvnpvthr': output.v_ppvnpvthr}
    connection.insert_one(row)
    

connection = connect()
# for output in named_product(v_i = [10, 50, 90], v_s = [12000], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], v_oap=[0.1, 0.2, 0.3], v_ppvnpvthr=[0.5]): 
# for output in named_product(v_i = [10], v_s = [12000], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.1, 0.3, 0.5, 0.7, 0.9], v_oap=[0.1, 0.15, 0.2, 0.25, 0.3], v_ppvnpvthr=[0.1, 0.5, 0.9]): 
for output in named_product(v_i = [10], v_s = [12000], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], v_oap=[0.1, 0.2, 0.3], v_ppvnpvthr=[0.5]): 
# for output in named_product(v_i = [10], v_s = [12000], v_mvp=[0.3], v_mbp=[1.0], v_oap=[0.1], v_ppvnpvthr=[0.5]): 

    threshold=output.v_i
    
    STEPS = output.v_s
    PPV_THR = output.v_ppvnpvthr
    NPV_THR = output.v_ppvnpvthr
    hitmax=False
    hitmin=False
    filename = "ce_db_0_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
    data = pd.read_csv('../../sampledata/'+filename, sep=',', error_bad_lines=False, encoding='latin1', header=0, nrows=3200001)
    STEPS = output.v_s
    starttime, endtime = [], []

    run_counts = 100
    for i in range (run_counts):
        interaction_number=1
        starttime.append(time.time())
        alllist = get_available_list(data)
        while True:    
            # samplelist = get_sample(3200000)
            # samplelist = get_available_list(data)
            samplelist=random.sample(alllist,50)
            evaluate(threshold, samplelist, data, interaction_number)
            step_records(i, interaction_number, threshold)
            # print(threshold)
    
            if (cases['recent'][0] + cases['recent'][1]) ==0:
                PPV=0
            else:
                PPV = cases['recent'][0] / (cases['recent'][0] + cases['recent'][1])
            if (cases['recent'][3] + cases['recent'][2])==0:
                NPV =0
            else:
                NPV = cases['recent'][3] / (cases['recent'][3] + cases['recent'][2])

            # if (PPV > PPV_THR and NPV > NPV_THR): #* 둘다 1에 가까우면 움직이지마!
            #     pass
            # elif(NPV < NPV_THR): #* 둘중 하나를 고쳐야되면 NPV부터 고쳐봐. 
            #     if threshold + 5 < 100:
            #         threshold+=5
            # if(PPV < PPV_THR):
            #     if threshold - 5 > 0:
            #         threshold-=5
            
            # if NPV > 0.95:
            #     threshold+=5
            #     if threshold>100:
            #         threshold=100
            # elif PPV < 0.95:
            #     threshold-=5
            #     if threshold<0:
            #         threshold=0
                
            #! 방법 3
            # if (PPV > PPV_THR and NPV > NPV_THR): #* 둘다 1에 가까우면 움직이지마!
            #     pass 
            # elif hitmax or hitmin: #* max, min 찍으면 반대방향으로 이동해.
            #     if hitmax:
            #         if threshold - 5 > 0:
            #             threshold-=5
            #     if hitmin:
            #         if threshold + 5 < 100:
            #             threshold+=5
            
            # elif(NPV < NPV_THR): #* 둘중 하나를 고쳐야되면 NPV부터 고쳐봐. 
            #     if threshold + 5 < 100:
            #         threshold+=5
                    
            # elif(PPV < PPV_THR):
            #     if threshold - 5 > 0:
            #         threshold-=5

            # if threshold == 95:
            #     hitmax=True
            #     hitmin=False
            # if threshold == 5:
            #     hitmin=True
            #     hitmax=False

            #! JH's approach
            # if (PPV>PPV_THR and NPV>NPV_THR):
            #     pass
            # elif (PPV>PPV_THR and NPV<NPV_THR):
            #     # if threshold + 5 < 100:
            #     #     threshold+=5
            #     if threshold - 5 > 0:
            #         threshold-=5
            # elif (NPV>NPV_THR and PPV<PPV_THR):
            #     # if threshold - 5 > 0:
            #     #     threshold-=5
            #     if threshold + 5 < 100:
            #         threshold+=5

            #! SK's approach
            # if (PPV>PPV_THR and NPV>NPV_THR):
            #     pass
            # elif (NPV<NPV_THR):
            #     if threshold - 5 > 0:
            #         threshold-= 5
            # elif (PPV<PPV_THR):
            #     if threshold + 5 < 100:
            #         threshold+=5
            # print("#:{}, THR: {}, PPV: {}, NPV: {}, ACC: {}".format(interaction_number,  threshold, PPV, NPV, (cases["recent"][0] + cases["recent"][3])/(cases["recent"][0] + cases["recent"][1] + cases["recent"][2] + cases["recent"][3]) *100))#accuracy[i][-1]))
            #! JH's 2nd approach 
            delta=1
            if cases['recent'][1] / (cases['recent'][0] + cases['recent'][1] + cases['recent'][2] + cases['recent'][3]) > 0.05:
                if threshold - delta >= 0:
                    threshold -= delta
            elif cases['recent'][2] / (cases['recent'][0] + cases['recent'][1] + cases['recent'][2] + cases['recent'][3]) > 0.05:
                if threshold + delta <= 100:
                    threshold += delta
            # print("#: ", interaction_number)
            # print("FP/all:", cases['recent'][1] / (cases['recent'][0] + cases['recent'][1] + cases['recent'][2] + cases['recent'][3]))
            # print("FN/all:", cases['recent'][2] / (cases['recent'][0] + cases['recent'][1] + cases['recent'][2] + cases['recent'][3]))
            # print("Thr:", threshold)
            # print("Acc:", (cases["gt"][0] + cases["gt"][3])/(cases["gt"][0] + cases["gt"][1] + cases["gt"][2] + cases["gt"][3]) *100)
            # print("Rec:", (cases["gt"][0])/(cases["gt"][0] + cases["gt"][2]) *100)


            if interaction_number == (STEPS)/100:
                print("run count: {} finished ".format(i))
                print("dtt {}".format(threshold))
                print("Accuracy {}".format(cum_accuracy[i][-1]))          
                threshold=output.v_i
                endtime.append(time.time())
        
                break
            
            interaction_number+=1
        cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
        result_values = defaultdict(lambda:[0,0,0,0]) #* precision, accuracy, recall
    
    avgsimtime=0
    errors=[]
    for r in range(run_counts):
        # print(r)
        avgsimtime+=endtime[r]-starttime[r]
        errors.append(endtime[r]-starttime[r])
    print("avgsimtime: ",avgsimtime/run_counts)
    print(statistics.stdev(errors))

    save_avg_accuracy(run_counts, output, int((STEPS+1)/100))
    cum_accuracy = defaultdict(list)
    step_accuracy = defaultdict(list)
    precision = defaultdict(list)
    recall = defaultdict(list)
    f1score = defaultdict(list)
    average_dtt= defaultdict(list)
slacknoti("RTMD finished", 's')
