import random
from copy import deepcopy

import pandas as pd
import numpy as np

MALICIOUS_BEHAVIOR_PROBABILITIES = [0.7, 0.9]
#MALICIOUS_BEHAVIOR_PROBABILITIES = [0.1, 0.2, 0.3, 0.4, 0.5]
PROB_ATTACKS = [0.1, 0.15, 0.2, 0.25, 0.3]
#PROB_ATTACKS = [0.1, 0.15, 0.2, 0.25, 0.3]
MALICIOUS_VEHICLE_PORTIONS = [0.1, 0.2, 0.3, 0.4]
#MALICIOUS_VEHICLE_PORTIONS = [0.1, 0.2, 0.3, 0.4]

# MALICIOUS_BEHAVIOR_PROBABILITIES = [0.1, 0.5]
# PROB_ATTACKS = [0.1]
# MALICIOUS_VEHICLE_PORTIONS = [0.2]

#MALICIOUS_BEHAVIOR_PROBABILITIES = [0.5]
#PROB_ATTACKS = [0.2]
#MALICIOUS_VEHICLE_PORTIONS = [0.2]

warmup_iterations = 5
MALICIOUS_STATUS = 1
BENIGN_STATUS = 0
NUM_ID = 500
NUM_USING_ID = 300
NUM_DE = 20000
NUM_WITH_NO_DIRECT_EVIDENCE = 100
NUM_DATA_PER_CONTEXT = 500
NUM_OPTIMAL_FRIENDS = 7
NUM_INTERACTION_BEHAVIOR = 600
DIRECT_EVIDENCE_WEIGHT = 0.5
PPV_THRESHHOLD = 0.9
NPV_THRESHHOLD = 0.45
THRESHHOLD_STEP = 0.05
NUM_SIMULATIONS = 1

class ID:
    def __init__(self, index, mvp):
        binary = [0, 1]
        weather_weight = [0.26, 0.74]
        visibility_weight = [0.493, 0.507]
        rush_hour_weight = [0.385, 0.615]
        gender_weight = [0.755, 0.245]
        age_weight = [0.802, 0.198]
        passenger_weight = [0.411, 0.589]
        self.weather = random.choices(binary, weather_weight)[0]
        self.visibility = random.choices(binary, visibility_weight)[0]
        self.rush_hour = random.choices(binary, rush_hour_weight)[0]
        self.gender = random.choices(binary, gender_weight)[0]
        self.age = random.choices(binary, age_weight)[0]
        self.passenger = random.choices(binary, passenger_weight)[0]

        self.index = index
        self.indirect_trust_value = 0
        self.personal_bias = random.uniform(-0.025, 0.025)
        self.environmental_bias = []
        self.history_good = []
        self.history_bad = []
        self.status = MALICIOUS_STATUS if random.uniform(0, 1) < mvp else BENIGN_STATUS
        for a in range(2):
            visibility_bias = []
            visibility_good = []
            visibility_bad = []
            for b in range(2):
                rush_hour_bias = []
                rush_hour_good = []
                rush_hour_bad = []
                for c in range(2):
                    gender_bias = []
                    gender_good = []
                    gender_bad = []
                    for d in range(2):
                        age_bias = []
                        age_good = []
                        age_bad = []
                        for e in range(2):
                            passenger_bias = []
                            passenger_good = []
                            passenger_bad = []
                            for f in range(2):
                                passenger_bias.append(random.uniform(-0.08, 0.08))
                                passenger_good.append(0)
                                passenger_bad.append(0)
                            age_bias.append(passenger_bias)
                            age_good.append(passenger_good)
                            age_bad.append(passenger_bad)
                        gender_bias.append(age_bias)
                        gender_good.append(age_good)
                        gender_bad.append(age_bad)
                    rush_hour_bias.append(gender_bias)
                    rush_hour_good.append(gender_good)
                    rush_hour_bad.append(gender_bad)
                visibility_bias.append(rush_hour_bias)
                visibility_good.append(rush_hour_good)
                visibility_bad.append(rush_hour_bad)
            self.environmental_bias.append(visibility_bias)
            self.history_good.append(visibility_good)
            self.history_bad.append(visibility_bad)


class ISharingFriend:
    def __init__(self, similarity_value, index):
        self.similarity_value = similarity_value
        self.index = index

    def __eq__(self, other):
        return self.similarity_value == other.similarity_value and self.index == other.index

    def __lt__(self, other):
        return self.similarity_value < other.similarity_value


def get_base_threshhold(current_id_db, weather, visibility, rush_hour, gender, age, passenger):
    tv_sum = 0
    same_context_count = 0

    for i in range(NUM_USING_ID):
        malicious_count = current_id_db[i].history_bad[weather][visibility][rush_hour][gender][age][passenger]
        benign_count = current_id_db[i].history_good[weather][visibility][rush_hour][gender][age][passenger]
        if benign_count + malicious_count == 0:
            continue
        else:
            same_context_count += 1
            tv_sum += benign_count / (benign_count + malicious_count)

    return tv_sum / same_context_count


def get_indirect_trust_value(i, ce_db, rsu_db):
    closest_friends = get_sorted_i_sharing_friends(ce_db, i)
    indirect_trust = 0
    close_friend_with_context_count = 0
    for isf in closest_friends:
        isf_id = rsu_db[isf.index]
        weather = isf_id.weather
        visibility = isf_id.visibility
        rush_hour = isf_id.rush_hour
        gender = isf_id.gender
        age = isf_id.age
        passenger = isf_id.passenger

        good = isf_id.history_good[weather][visibility][rush_hour][gender][age][passenger]
        bad = isf_id.history_bad[weather][visibility][rush_hour][gender][age][passenger]
        if good + bad > 0:
            close_friend_with_context_count += 1
            indirect_trust += isf.similarity_value * good / (good + bad)
            if close_friend_with_context_count == NUM_OPTIMAL_FRIENDS:
                break
    return indirect_trust / close_friend_with_context_count


def get_sorted_i_sharing_friends(ce_db, i):
    base_id = ce_db[i]
    i_sharing_friends = []
    for j in range(NUM_ID):
        if i == j:
            continue
        sum_direct_trust_value = 0
        context_count = 0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        for e in range(2):
                            for f in range(2):
                                base_good = base_id.history_good[a][b][c][d][e][f]
                                base_bad = base_id.history_bad[a][b][c][d][e][f]
                                isf_good = ce_db[j].history_good[a][b][c][d][e][f]
                                isf_bad = ce_db[j].history_bad[a][b][c][d][e][f]
                                if base_good + base_bad == 0 or isf_good + isf_bad == 0:
                                    continue
                                else:
                                    context_count += 1
                                    sum_direct_trust_value += \
                                        abs(base_good / (base_good + base_bad) - isf_good / (isf_good + isf_bad))
        if context_count == 0:
            i_sharing_friends.append(ISharingFriend(0, j))
        else:
            similarity_value = 1 - (sum_direct_trust_value / context_count)
            i_sharing_friends.append(ISharingFriend(similarity_value, j))
    return sorted(i_sharing_friends, reverse=True)


def get_indirect_trust_values(ce_db, rsu_db):
    for i in range(NUM_WITH_NO_DIRECT_EVIDENCE):
        rsu_db[i].indirect_trust_value = get_indirect_trust_value(i, ce_db, rsu_db,)


def iterate_interactions(current_id_db, threshhold, mbp, oap, mvp, iteration_num, RSU_MBP_PER_ENV_CONTEXT):
    accuracy_sum = 0
    accuracy_per_iteration = []

    direct_tvs = []
    indirect_tvs = []
    statuses = []

    weather_column = []
    visibility_column = []
    rush_hour_column = []
    gender_column = []
    age_column = []
    passengers_column = []
    good_history_column = []
    bad_history_column = []
    tv_column = []
    decision_column = []
    status_column = []

    for interaction in range(NUM_INTERACTION_BEHAVIOR):
        tp = 0  # Correctly detected malicious
        tn = 0  # Correctly detected benign
        fp = 0  # Incorrectly detected malicious
        fn = 0  # Incorrectly detected benign

        for interaction_vehicle_index in range(NUM_WITH_NO_DIRECT_EVIDENCE):
            interaction_id = current_id_db[interaction_vehicle_index]
            weather = interaction_id.weather
            visibility = interaction_id.visibility
            rush_hour = interaction_id.rush_hour
            gender = interaction_id.gender
            age = interaction_id.age
            passenger = interaction_id.passenger
            malicious_prob = get_malicious_behavior_probability(
                interaction_id, weather, visibility, rush_hour, gender, age, passenger, RSU_MBP_PER_ENV_CONTEXT)
            direct_good = interaction_id.history_good[weather][visibility][rush_hour][gender][age][passenger]
            direct_bad = interaction_id.history_bad[weather][visibility][rush_hour][gender][age][passenger]

            if random.uniform(0, 1) < malicious_prob:
                behavior = MALICIOUS_STATUS
            else:
                behavior = BENIGN_STATUS

            if random.uniform(0, 1) < oap:
                behavior = 1 - behavior

            if behavior == MALICIOUS_STATUS:
                interaction_id.history_bad[weather][visibility][rush_hour][gender][age][passenger] += 1
            else:
                interaction_id.history_good[weather][visibility][rush_hour][gender][age][passenger] += 1

            weather_column.append(weather)
            visibility_column.append(visibility)
            rush_hour_column.append(rush_hour)
            gender_column.append(gender)
            age_column.append(age)
            passengers_column.append(passenger)
            good_history_column.append(direct_good)
            bad_history_column.append(direct_bad)
            status_column.append(interaction_id.status)
            direct_count = direct_good + direct_bad
            if interaction == 0:
                direct_tv = 0.5
            else:
                direct_tv = direct_good / direct_count

            indirect_tv = interaction_id.indirect_trust_value
            beta = direct_count / (60 + direct_count)
            tv = direct_tv * beta + indirect_tv * (1 - beta)

            tv_column.append(tv)
            direct_tvs.append(direct_tv)
            indirect_tvs.append(indirect_tv)
            statuses.append(interaction_id.status)

            if tv < threshhold:
                decision = MALICIOUS_STATUS
            else:
                decision = BENIGN_STATUS

            decision_column.append(decision)
            if decision == interaction_id.status:
                if decision == BENIGN_STATUS:
                    tn += 1
                else:
                    tp += 1
            else:
                if decision == BENIGN_STATUS:
                    # print("FN for TV: ", tv)
                    # print("Direct Bad: ", direct_bad)
                    # print("Direct Good: ", direct_good)
                    fn += 1
                else:
                    # print("FP for TV: ", tv)
                    # print("Direct Bad: ", direct_bad)
                    # print("Direct Good: ", direct_good)
                    fp += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        # print("interaction:", interaction)
        # print("threshold:", threshhold)
        # print("tp: ", tp)
        # print("tn: ", tn)
        # print("fp: ", fp)
        # print("fn: ", fn)

        # print(accuracy)
        accuracy_sum += accuracy
        accuracy_per_iteration.append(accuracy)

        # if interaction % 2 == 0:
        total_positive = tp + fp
        if total_positive != 0:
            ppv = tp / total_positive
            if ppv < PPV_THRESHHOLD:
                threshhold -= THRESHHOLD_STEP
                if threshhold < 0:
                    threshhold = 0
        total_negative = tn + fn
        if total_negative != 0:
            npv = tn / total_negative
            if npv < NPV_THRESHHOLD:
                threshhold += THRESHHOLD_STEP
                if threshhold > 1:
                    threshhold = 1

    rl_df = pd.DataFrame({'direct_tv': np.array(np.asarray(direct_tvs)),
                          'indirect_tv': np.array(np.asarray(indirect_tvs)),
                          'status': np.asarray(np.asarray(statuses))})
    rl_df.to_csv('is_df_' + str(iteration_num) + '_' + str(mbp) + 'mbp' + str(oap) + 'oap' + str(mvp) + 'mvp.csv',
                 index=False)


def get_env_context_index(weather, visibility, rush_hour, gender, age, passenger):
    return int(str(weather) + str(visibility) + str(rush_hour) + str(gender) + str(age) + str(passenger), 2)


def initialize_env_mbp(mbp):
    CE_MBP_PER_ENV_CONTEXT = []
    RSU_MBP_PER_ENV_CONTEXT = []
    for i in range(64):
        CE_MBP_PER_ENV_CONTEXT.append(mbp + random.uniform(-0.10, 0.10))
        RSU_MBP_PER_ENV_CONTEXT.append(mbp + random.uniform(-0.10, 0.10))
    return CE_MBP_PER_ENV_CONTEXT, RSU_MBP_PER_ENV_CONTEXT


def simulate_behavior(mbp, oap, mvp, iteration_num):
    # create history database of the CE
    ce_db = initialize_id_database(mvp)

    CE_MBP_PER_ENV_CONTEXT, RSU_MBP_PER_ENV_CONTEXT = initialize_env_mbp(mbp)
    current_id_db = deepcopy(ce_db)
    data_creation(ce_db, 0, NUM_ID, oap, CE_MBP_PER_ENV_CONTEXT)
    data_creation(current_id_db, NUM_WITH_NO_DIRECT_EVIDENCE, NUM_ID, oap, RSU_MBP_PER_ENV_CONTEXT)
    get_indirect_trust_values(ce_db, current_id_db)
    base_threshhold = 0.5
    iterate_interactions(current_id_db, base_threshhold, mbp, oap, mvp, iteration_num, RSU_MBP_PER_ENV_CONTEXT)


def initialize_id_database(mvp):
    id_db = []
    for i in range(NUM_ID):
        id_db.append(ID(i, mvp))
    return id_db


def data_creation(db, start, end, oap, env_mbps):
    weather_column = []
    visibility_column = []
    rush_hour_column = []
    gender_column = []
    age_column = []
    passengers_column = []
    good_history_column = []
    bad_history_column = []
    behavior_column = []
    status_column = []

    for index in range(start, end - 1):
        used_id = db[index]
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        for e in range(2):
                            for f in range(2):
                                malicious_prob = get_malicious_behavior_probability(used_id, a, b, c, d, e, f, env_mbps)
                                for g in range(NUM_DATA_PER_CONTEXT):
                                    if random.uniform(0, 1) < malicious_prob:
                                        behavior = MALICIOUS_STATUS
                                    else:
                                        behavior = BENIGN_STATUS

                                    if random.uniform(0, 1) < oap:
                                        behavior = 1 - behavior

                                    if behavior == MALICIOUS_STATUS:
                                        used_id.history_bad[a][b][c][d][e][f] += 1
                                    else:
                                        used_id.history_good[a][b][c][d][e][f] += 1
                                    weather_column.append(a)
                                    visibility_column.append(b)
                                    rush_hour_column.append(c)
                                    gender_column.append(d)
                                    age_column.append(e)
                                    passengers_column.append(f)
                                    good_history_column.append(used_id.history_good[a][b][c][d][e][f])
                                    bad_history_column.append(used_id.history_bad[a][b][c][d][e][f])
                                    behavior_column.append(behavior)
                                    status_column.append(used_id.status)
    df = pd.DataFrame(
        {
            'weather': np.array(np.asarray(weather_column)),
            'visibility': np.array(np.asarray(visibility_column)),
            'rush_hour': np.array(np.asarray(rush_hour_column)),
            'gender': np.array(np.asarray(gender_column)),
            'age': np.array(np.asarray(age_column)),
            'passenger': np.array(np.asarray(passengers_column)),
            'good_history': np.array(np.asarray(good_history_column)),
            'bad_history': np.array(np.asarray(bad_history_column)),
            'behavior': np.array(np.asarray(behavior_column)),
            'status': np.array(np.asarray(status_column)),
        })
    return df


def get_malicious_behavior_probability(id_obj, weather, visibility, rush_hour, gender, age,
                                       passenger, mbp_per_context):
    if id_obj.status == BENIGN_STATUS:
        malicious_prob = \
            id_obj.personal_bias + id_obj.environmental_bias[weather][visibility][rush_hour][gender][age][passenger]
        if malicious_prob < 0:
            return 0
        else:
            return random.uniform(0, malicious_prob)

    # probability to display benign behavior when bad/good, rush/not_rush, male/female, under_35/over_45, with/without
    # TODO Try increasing the malicious probability (increase so that it has a clearer difference with the

    return mbp_per_context[get_env_context_index(weather, visibility, rush_hour, gender, age, passenger)] \
           + id_obj.personal_bias + id_obj.environmental_bias[weather][visibility][rush_hour][gender][age][passenger]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for iteration in range(NUM_SIMULATIONS):
        for malicious_behavior_probability in MALICIOUS_BEHAVIOR_PROBABILITIES:
            for outside_attack_probability in PROB_ATTACKS:
                for malicious_vehicle_portion in MALICIOUS_VEHICLE_PORTIONS:
                    print("simulating behavior for mbp: ", malicious_behavior_probability,
                          "oap: ", outside_attack_probability,
                          "mvp: ", malicious_vehicle_portion)
                    simulate_behavior(malicious_behavior_probability,
                                      outside_attack_probability,
                                      malicious_vehicle_portion,
                                      iteration)
