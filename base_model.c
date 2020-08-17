#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#define BIAS_RANGE 160
#define PERSONAL_RANGE 50
//#define MODIFICATION_PROB 200

//#define I_WEIGHT 0.3
//#define OB_WEIGHT 0.3

#define PROB_MAL 0.20

#define MAX_NUM_ID 2000

#define MAL_UNKNOWN_PORTION 0.20

#define NUMBER_INTERACTION_VEHICLE 50
#define NUMBER_INTERACTOIN_BEHAVIOR 720
#define NUMBER_BEHAVIOR 240

#define NUM_I_SHARING_FRIENDS 7

#define SIM_THRESH 0.90
#define I_SIM_THRESH 0.80

#define CONTEXT_SHARING_THRESH 1
#define ESTIMATION_INTERACTION_THRESH 1

#define NO_HISTORY -1
#define NOT_I_SHARED -1

#define NUM_CONTEXT_FEATURE 6

#define BENIGN_STATUS 1
#define MALICIOUS_STATUS 0

#define BAD_WEATHER 0
#define GOOD_WEATHER 1

#define BAD_VISIBILITY 0
#define GOOD_VISIBILITY 1

#define RUSH_HOUR 0
#define NOT_RUSH_HOUR 1

typedef struct data
{
	int id;

	int weather;
	int visibility;
	int rush_hour;

	int good;
	int bad;
}data;
typedef struct ID_data
{
	int id;
	int status;

	int good;
	int bad;

	int context_good[2][2][2];
	int context_bad[2][2][2];

	// trust value
	double trust_value;

	// bias value
	double bias_personal;
	double bias_context[2][2][2];
}ID_data;
typedef struct probability
{
	double weather[2];
	double visibility[2];
	double rush_hour[2];
}probability;

void generate_probability();
double plus_minus_five_random();

void estimate_trust_value(ID_data **ID_data_base, int num_ID);
void id_generation(ID_data **ID_data_base, int num_ID);
void copy_ID_data(ID_data *base_ID_data_base, ID_data **target_ID_data_base, int num_ID);

void data_generation(data **data_base, ID_data **ID_data_base, int num_using_ID, int num_data, int num_ID);
int behavior_simulation(int weather, int visibility, int rush_hour, ID_data *target_ID);
double behavior_probability(int weather, int visibility, int rush_hour, ID_data *target_ID);

double* I_similartiy_estimation(ID_data *ID_data_base, int ID_trustee, int num_ID);
int* similarity_sorting(double* similarity_list, int num_ID);

double estimate_temp_trust(ID_data *ID_data_base, double* similarity_list, int target_ID, int weather, int visibility, int rush_hour, int num_ID);
double estimate_initial_trust(ID_data *ID_data_base, double* similarity_list, int target_ID, int weather, int visibility, int rush_hour, int num_ID, int *sorted_sim_list);
void trust_simulation(int num_ID, int num_controller_ID_data, int num_neighbor_ID_data, int num_data, double weight, double threshold, int prob_forge, double mal_vehicle_portion, FILE *fp, double *I_trust_result, double *ob_trust_result, double *none_result);

void simulation_function(int num_ID, int num_controller_ID_data, int num_neighbor_ID_data, int num_data, double weight, double threshold, int prob_forge, double mal_vehicle_portion, FILE *fp);

probability *probability_base;

int main()
{
	int i, j, k;

	double weight[] = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 };
	double threshold[] = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 };
	int prob_forge[] = { 0, 100, 200, 300 };

	double mal_vehicle_portion[] = { 0, 0.1, 0.2, 0.3 };

	FILE *fp;

	srand(time(NULL));
	// generate_probability();

	// // open file
	// fp = fopen("detection_accuracy_dynamic_RTM.txt", "w");
	
	// simulation start
	printf("SIMULATION START\n");
	for (j = 1; j < 2; j++)
	{	
		for (i = 5; i < 6; i++)
		{
			for(k = 5; k < 6; k++) {
				generate_probability();
				char buf[12];
				snprintf(buf, 12, "data_%d_.txt", k);
				// open file
				fp = fopen(buf, "a");
				fprintf(fp, "bad\tgood\trush_hour\tvisibility\tweather\tIS_Good\tIS_Bad\tI_trust_val\tdecision_thresh\tbad_weather\tgood_weather\tbad_visibility\tgood_visibility\trush_hour\tnot_rush_hour\tI_decision\tactual_status\n");
				simulation_function(500, 300, 500, 50000, weight[i], threshold[5], prob_forge[j], mal_vehicle_portion[2], fp);
				fclose(fp);
			}
		}
	}
	printf("SIMULATION DONE\n");

	// close file
	fclose(fp);

	// free memory
	free(probability_base);

	return 0;
}

void generate_probability()
{
	probability_base = (probability*)malloc(sizeof(probability));

	// probability_base->weather[BAD_WEATHER] = 0.797297 + plus_minus_five_random() / (double)100000;
	// probability_base->weather[GOOD_WEATHER] = 0.923077 + plus_minus_five_random() / (double)100000;

	// probability_base->visibility[BAD_VISIBILITY] = 0.943205 + plus_minus_five_random() / (double)100000;
	// probability_base->visibility[GOOD_VISIBILITY] = 0.867850 + plus_minus_five_random() / (double)100000;

	// probability_base->rush_hour[RUSH_HOUR] = 0.935065 + plus_minus_five_random() / (double)100000;
	// probability_base->rush_hour[NOT_RUSH_HOUR] = 0.804878 + plus_minus_five_random() / (double)100000;

	probability_base->weather[BAD_WEATHER] = 0.797297;
	probability_base->weather[GOOD_WEATHER] = 0.923077;

	probability_base->visibility[BAD_VISIBILITY] = 0.943205;
	probability_base->visibility[GOOD_VISIBILITY] = 0.867850 ;

	probability_base->rush_hour[RUSH_HOUR] = 0.935065;
	probability_base->rush_hour[NOT_RUSH_HOUR] = 0.804878;

	//probability_base->weather[BAD_WEATHER] = 0.797297;
	//probability_base->weather[GOOD_WEATHER] = 0.923077;

	//probability_base->visibility[BAD_VISIBILITY] = 0.943205;
	//probability_base->visibility[GOOD_VISIBILITY] = 0.867850;

	//probability_base->rush_hour[RUSH_HOUR] = 0.935065;
	//probability_base->rush_hour[NOT_RUSH_HOUR] = 0.804878;
}

// double plus_minus_five_random()
// {
	
// 	int num = rand() % 10001 - 5000; 
// 	printf("rand var : %d ", num); 
// 	return num;
        
// }

void estimate_trust_value(ID_data **ID_data_base, int num_ID)
{
	int i;

	for (i = 0; i < num_ID; i++)
	{
		if ((*ID_data_base)[i].good == 0 && (*ID_data_base)[i].bad == 0) (*ID_data_base)[i].trust_value = NO_HISTORY;
		else (*ID_data_base)[i].trust_value = (*ID_data_base)[i].good / (double)((*ID_data_base)[i].good + (*ID_data_base)[i].bad);
	}
}
void id_generation(ID_data **ID_data_base, int num_ID)
{
	int i, j, k, l, r;

	ID_data *new_IDs;

	(*ID_data_base) = (ID_data*)malloc(sizeof(ID_data) * num_ID);
	new_IDs = (*ID_data_base);

	for (i = 0; i < num_ID; i++)
	{
		new_IDs[i].id = i;

		new_IDs[i].good = 0;
		new_IDs[i].bad = 0;

		new_IDs[i].trust_value = NO_HISTORY;

		for (j = 0; j < 2; j++)
		{
			for (k = 0; k < 2; k++)
			{
				for (l = 0; l < 2; l++)
				{
					new_IDs[i].context_good[j][k][l] = 0;
					new_IDs[i].context_bad[j][k][l] = 0;
				}
			}
		}
	}

	for (i = 0; i < num_ID; i++)
	{
		r = rand() % 1000;
		if (r < PROB_MAL * 1000) new_IDs[i].status = MALICIOUS_STATUS;
		else new_IDs[i].status = BENIGN_STATUS;

		for (j = 0; j < 2; j++)
		{
			for (k = 0; k < 2; k++)
			{
				for (l = 0; l < 2; l++)
				{
					new_IDs[i].bias_context[j][k][l] = ((rand() % (BIAS_RANGE + 1)) - (BIAS_RANGE / 2)) / (double)1000;
				}
			}
		}
		new_IDs[i].bias_personal = (rand() % (PERSONAL_RANGE + 1)) / (double)1000;
	}
}
void copy_ID_data(ID_data *base_ID_data_base, ID_data **target_ID_data_base, int num_ID)
{
	int i, j, k, l;
	ID_data *temp_ptr;

	(*target_ID_data_base) = (ID_data*)malloc(sizeof(ID_data) * num_ID);
	temp_ptr = (*target_ID_data_base);

	for (i = 0; i < num_ID; i++)
	{
		// features of ID
		temp_ptr[i].id = base_ID_data_base[i].id;
		temp_ptr[i].status = base_ID_data_base[i].status;

		temp_ptr[i].good = base_ID_data_base[i].good;
		temp_ptr[i].bad = base_ID_data_base[i].bad;

		temp_ptr[i].trust_value = base_ID_data_base[i].trust_value;

		for (j = 0; j < 2; j++)
		{
			for (k = 0; k < 2; k++)
			{
				for (l = 0; l < 2; l++)
				{
					temp_ptr[i].context_good[j][k][l] = base_ID_data_base[i].context_good[j][k][l];
					temp_ptr[i].context_bad[j][k][l] = base_ID_data_base[i].context_bad[j][k][l];
				}
			}
		}

		for (j = 0; j < 2; j++)
		{
			for (k = 0; k < 2; k++)
			{
				for (l = 0; l < 2; l++)
				{
					temp_ptr[i].bias_context[j][k][l] = ((rand() % (BIAS_RANGE + 1)) - (BIAS_RANGE / 2)) / (double)1000;
				}
			}
		}
		temp_ptr[i].bias_personal = (rand() % (PERSONAL_RANGE + 1)) / (double)1000;
	}
}

void data_generation(data **data_base, ID_data **ID_data_base, int num_using_ID, int num_data, int num_ID)
{
	int i, j, r, is_in;
	int id, weather, visibility, rush_hour;
	double temp_probability;

	data *new_data_base;
	int *random_IDs = NULL;

	// select ID randomly
	if (num_using_ID < num_ID)
	{
		random_IDs = (int*)malloc(sizeof(int) * num_using_ID);

		for (i = 0; i < num_using_ID; i++)
		{
			do
			{
				r = rand() % num_ID;

				is_in = 0;
				for (j = 0; j < i; j++)
				{
					if (random_IDs[j] == r)
					{
						is_in = 1;
						break;
					}
				}
			} while (is_in);

			random_IDs[i] = r;
		}
	}

	// allocate memory for database
	(*data_base) = (data*)malloc(sizeof(data) * num_data);
	new_data_base = (*data_base);

	// data generation
	for (i = 0; i < num_data; i++)
	{
		// select id randomly
		if (num_using_ID < num_ID) id = random_IDs[rand() % num_using_ID];
		else id = rand() % num_ID;

		new_data_base[i].id = id;

		// set weather, visibility, and rush hour
		r = rand() % 1000;
		if (r < 260) weather = BAD_WEATHER;
		else weather = GOOD_WEATHER;

		r = rand() % 1000;
		if (r < 493) visibility = BAD_VISIBILITY;
		else visibility = GOOD_VISIBILITY;

		r = rand() % 1000;
		if (r < 385) rush_hour = RUSH_HOUR;
		else rush_hour = NOT_RUSH_HOUR;

		// set weather, visibility, and rush hour to data base
		new_data_base[i].weather = weather;
		new_data_base[i].visibility = visibility;
		new_data_base[i].rush_hour = rush_hour;

		// behavior hitory initialization
		new_data_base[i].good = 0;
		new_data_base[i].bad = 0;

		// set number of behaviors randomly
		temp_probability = behavior_probability(weather, visibility, rush_hour, &((*ID_data_base)[id]));
		temp_probability = temp_probability * 1000;
		for (j = 0; j < NUMBER_BEHAVIOR; j++)
		{
			r = rand() % 1000;

			if (r < temp_probability)
			{
				(*ID_data_base)[id].bad++;
				(*ID_data_base)[id].context_bad[weather][visibility][rush_hour]++;
				new_data_base[i].bad++;
			}
			else
			{
				(*ID_data_base)[id].good++;
				(*ID_data_base)[id].context_good[weather][visibility][rush_hour]++;
				new_data_base[i].good++;
			}
		}
	}

	// estimate trust value
	estimate_trust_value(ID_data_base, num_ID);

	// free memory
	if (num_using_ID < num_ID) free(random_IDs);
}
int behavior_simulation(int weather, int visibility, int rush_hour, ID_data *target_ID)
{
	int r;
	double malicious_probability;
	//double context_probability[3];

	// CONTEXT PROBABILITY
	if (target_ID->status == BENIGN_STATUS)
	{
		malicious_probability = ((rand() % (PERSONAL_RANGE + 1)) / (double)1000);
	}
	else if (target_ID->status == MALICIOUS_STATUS)
	{
		//context_probability[0] = probability_base->weather[weather];
		//context_probability[1] = probability_base->visibility[visibility];
		//context_probability[2] = probability_base->rush_hour[rush_hour];

		//malicious_probability = (0.340 * context_probability[0]) + (0.318 * context_probability[1]) + (0.342 * context_probability[2]) - target_ID->bias_personal;

		malicious_probability = probability_base->weather[weather] * probability_base->visibility[visibility] * probability_base->rush_hour[rush_hour] + target_ID->bias_context[weather][visibility][rush_hour] - ((rand() % (PERSONAL_RANGE + 1)) / (double)1000);
	}

	malicious_probability = malicious_probability * 1000;

	r = rand() % 1000;

	if (r < malicious_probability) return MALICIOUS_STATUS;
	else return BENIGN_STATUS;
}
double behavior_probability(int weather, int visibility, int rush_hour, ID_data *target_ID)
{
	double malicious_probability;
	//double context_probability[3];

	// CONTEXT PROBABILITY
	if (target_ID->status == BENIGN_STATUS)
	{
		malicious_probability = ((rand() % (PERSONAL_RANGE + 1)) / (double)1000);
	}
	else if (target_ID->status == MALICIOUS_STATUS)
	{
		//context_probability[0] = probability_base->weather[weather];
		//context_probability[1] = probability_base->visibility[visibility];
		//context_probability[2] = probability_base->rush_hour[rush_hour];

		//malicious_probability = (0.340 * context_probability[0]) + (0.318 * context_probability[1]) + (0.342 * context_probability[2]) - target_ID->bias_personal;

		malicious_probability = probability_base->weather[weather] * probability_base->visibility[visibility] * probability_base->rush_hour[rush_hour] + target_ID->bias_context[weather][visibility][rush_hour] - ((rand() % (PERSONAL_RANGE + 1)) / (double)1000);
	}

	return malicious_probability;
}

double* I_similartiy_estimation(ID_data *ID_data_base, int ID_trustee, int num_ID)
{
	int i, j, k, l;
	int num_sharing_context;
	double diff_sum;

	double tv_trustee[2][2][2];
	double tv_reference[MAX_NUM_ID][2][2][2];

	// allocate similarity list
	double* similarity_values = NULL;
	similarity_values = (double*)malloc(sizeof(double) * num_ID);

	// estimate trust value of trustee
	for (i = 0; i < 2; i++)
	{
		for (j = 0; j < 2; j++)
		{
			for (k = 0; k < 2; k++)
			{
				if (ID_data_base[ID_trustee].context_good[i][j][k] == 0 && ID_data_base[ID_trustee].context_bad[i][j][k] == 0) tv_trustee[i][j][k] = NO_HISTORY;
				else tv_trustee[i][j][k] = ID_data_base[ID_trustee].context_good[i][j][k] / (double)(ID_data_base[ID_trustee].context_good[i][j][k] + ID_data_base[ID_trustee].context_bad[i][j][k]);
			}
		}
	}

	// estimate trust value of reference vehicles
	for (l = 0; l < num_ID; l++)
	{
		if (l == ID_trustee) continue;

		for (i = 0; i < 2; i++)
		{
			for (j = 0; j < 2; j++)
			{
				for (k = 0; k < 2; k++)
				{
					if (ID_data_base[l].context_good[i][j][k] == 0 && ID_data_base[l].context_bad[i][j][k] == 0) tv_reference[l][i][j][k] = NO_HISTORY;
					else tv_reference[l][i][j][k] = ID_data_base[l].context_good[i][j][k] / (double)(ID_data_base[l].context_good[i][j][k] + ID_data_base[l].context_bad[i][j][k]);
				}
			}
		}
	}

	for (i = 0; i < num_ID; i++)
	{
		if (l == ID_trustee)
		{
			// trustee itself
			similarity_values[i] = NOT_I_SHARED;
			continue;
		}

		// compare trust value
		num_sharing_context = 0;
		diff_sum = 0;

		for (j = 0; j < 2; j++)
		{
			for (k = 0; k < 2; k++)
			{
				for (l = 0; l < 2; l++)
				{
					if (tv_trustee[j][k][l] >= 0 && tv_reference[i][j][k][l] >= 0)
					{
						diff_sum = diff_sum + fabs(tv_trustee[j][k][l] - tv_reference[i][j][k][l]);

						num_sharing_context++;
					}
				}
			}
		}

		if (num_sharing_context >= CONTEXT_SHARING_THRESH)
		{
			// estimate similarity
			similarity_values[i] = 1 - (diff_sum / num_sharing_context);
		}
		else
		{
			similarity_values[i] = NOT_I_SHARED;
		}

		// classify I-sharing friend with threshold
		if (similarity_values[i] < I_SIM_THRESH)
		{
			similarity_values[i] = NOT_I_SHARED;
		}
	}

	// return list of I-sharing values
	return similarity_values;
}
int* similarity_sorting(double* similarity_list, int num_ID)
{
	int i, j, temp;
	int* sorted_list = (int*)malloc(sizeof(int) * num_ID);

	for (i = 0; i < num_ID; i++) sorted_list[i] = i;

	for (i = 1; i < num_ID; i++)
	{
		for (j = 0; j < num_ID - i; j++)
		{
			if (similarity_list[sorted_list[j]] < similarity_list[sorted_list[j + 1]])
			{
				temp = sorted_list[j + 1];
				sorted_list[j + 1] = sorted_list[j];
				sorted_list[j] = temp;
			}
		}
	}

	return sorted_list;
}

double estimate_temp_trust(ID_data *ID_data_base, double* similarity_list, int target_ID, int weather, int visibility, int rush_hour, int num_ID)
{
	int i, leveraging_count;
	double trust_value;

	trust_value = 0;
	leveraging_count = 0;
	for (i = 0; i < num_ID; i++)
	{
		if (similarity_list[i] < 0 || i == target_ID || ID_data_base[i].trust_value < 0) continue;

		if (ID_data_base[i].context_good[weather][visibility][rush_hour] > 0 && ID_data_base[i].context_bad[weather][visibility][rush_hour] > 0)
		{
			trust_value = trust_value + (similarity_list[i] * ID_data_base[i].context_good[weather][visibility][rush_hour] / (double)(ID_data_base[i].context_good[weather][visibility][rush_hour] + ID_data_base[i].context_bad[weather][visibility][rush_hour]));
			leveraging_count++;
		}
	}

	if (leveraging_count > 0)
	{
		// estimate trust value
		trust_value = trust_value / leveraging_count;
		return trust_value;
	}
	else return NO_HISTORY;
}
double estimate_initial_trust(ID_data *ID_data_base, double* similarity_list, int target_ID, int weather, int visibility, int rush_hour, int num_ID, int *sorted_sim_list)
{
	int i, temp_id, leveraging_count;
	double trust_value;

	trust_value = 0;
	leveraging_count = 0;
	for (i = 0; i < num_ID; i++)
	{
		temp_id = sorted_sim_list[i];

		// check number of friends
		if (leveraging_count >= NUM_I_SHARING_FRIENDS) break;
		// No more I-sharing vehicles
		if (similarity_list[temp_id] < 0) break;

		// continue when history of ID number 'i' is absence
		if (ID_data_base[temp_id].trust_value < 0 || temp_id == target_ID) continue;

		if (ID_data_base[temp_id].context_good[weather][visibility][rush_hour] > 0 && ID_data_base[temp_id].context_bad[weather][visibility][rush_hour] > 0)
		{
			trust_value = trust_value + (similarity_list[temp_id] * ID_data_base[temp_id].context_good[weather][visibility][rush_hour] / (double)(ID_data_base[temp_id].context_good[weather][visibility][rush_hour] + ID_data_base[temp_id].context_bad[weather][visibility][rush_hour]));
			leveraging_count++;
		}
	}

	if (leveraging_count > 0)
	{
		// estimate trust value
		trust_value = trust_value / leveraging_count;
		return trust_value;
	}
	else return NO_HISTORY;
}
void trust_simulation(int num_ID, int num_controller_ID_data, int num_neighbor_ID_data, int num_data, double weight, double threshold, int prob_forge, double mal_vehicle_portion, FILE *fp, double *I_trust_result, double *ob_trust_result, double *none_result)
{

	// data base
	data *controller_data_base = NULL;
	ID_data *controller_ID_data_base = NULL;

	data *neighbor_a_data_base = NULL;
	data *neighbor_b_data_base = NULL;
	ID_data *neighbor_a_ID_data_base = NULL;
	ID_data *neighbor_b_ID_data_base = NULL;

	// interaction vehicle
	int *list_interaction = NULL;
	int *list_unknown = NULL;
	int *list_unknown_check_I_sharing = NULL;
	double *list_I_sharing_trust_value = NULL;
	double *list_objective_trust_value = NULL;

	// I-sharing list
	double* similarity_a = NULL;
	double* similarity_b = NULL;
	double** similarity_list = NULL;
	int *sorted_sim_list = NULL;

	double** interaction_sim_list = NULL;

	// history for belief
	int *I_sharing_num_good = NULL;
	int *I_sharing_num_bad = NULL;

	int *objective_num_good = NULL;
	int *objective_num_bad = NULL;

	// results
	int *list_behavior_results = NULL;
	double *list_trust_results = NULL;

	int I_sharing_decision, objective_decision;
	int I_TP, I_FP, I_FN, I_TN;
	int OB_TP, OB_FP, OB_FN, OB_TN;
	int NONE_true, NONE_false;

	int D_I_TP, D_I_FP, D_I_FN, D_I_TN;
	int D_OB_TP, D_OB_FP, D_OB_FN, D_OB_TN;

	// simulation
	int i, j, r, target_ID, reference_flag;
	int weather, visibility, rush_hour;

	// dynamic threshold
	int I_flag, ob_flag;
	int verification_count;
	double omega;
	
	double prev_PPV_I_sharing, prev_PPV_objective;
	double prev_NPV_I_sharing, prev_NPV_objective;

	// num of data and IDs
	int num_controller_data = num_data;
	int num_a_data = num_data;
	int num_b_data = num_data;

	int num_a_ID_data = num_neighbor_ID_data;
	int num_b_ID_data = num_neighbor_ID_data;

	// allocation
	int num_unknown;
	int num_benign_vehicle, num_mal_vehicle;

	// estimated trust value
	double objective_value_a, objective_value_b, objective_value;
	double I_sharing_direct, objective_direct;
	double I_sharing_tv, objective_tv;

	double temp_tv, temp_target_tv;
	int temp_count;

	// F1 score
	double I_recall, I_precision;
	double ob_recall, ob_precision;
	double none_recall, none_precision;

	// threshold
	double decision_thresh_I_sharing, decision_thresh_objective;

	// generate ID data base
	id_generation(&controller_ID_data_base, num_ID);
	copy_ID_data(controller_ID_data_base, &neighbor_a_ID_data_base, num_ID);
	copy_ID_data(controller_ID_data_base, &neighbor_b_ID_data_base, num_ID);

	// generate interaction data base
	data_generation(&controller_data_base, &controller_ID_data_base, num_controller_ID_data, num_controller_data, num_ID);
	data_generation(&neighbor_a_data_base, &neighbor_a_ID_data_base, num_a_ID_data, num_a_data, num_ID);
	data_generation(&neighbor_b_data_base, &neighbor_b_ID_data_base, num_b_ID_data, num_b_data, num_ID);

	// set weather, visibility, and rush hour
	r = rand() % 1000;
	if (r < 260) weather = BAD_WEATHER;
	else weather = GOOD_WEATHER;

	r = rand() % 1000;
	if (r < 493) visibility = BAD_VISIBILITY;
	else visibility = GOOD_VISIBILITY;

	r = rand() % 1000;
	if (r < 385) rush_hour = RUSH_HOUR;
	else rush_hour = NOT_RUSH_HOUR;

	// set threshold
	temp_tv = 0;
	temp_count = 0;
	for (i = 0; i < num_ID; i++)
	{
		if (controller_ID_data_base[i].status == MALICIOUS_STATUS && controller_ID_data_base[i].trust_value >= 0)
		{
			if(controller_ID_data_base[i].context_good[weather][visibility][rush_hour] != 0 || controller_ID_data_base[i].context_bad[weather][visibility][rush_hour] != 0)
			{
				temp_target_tv = controller_ID_data_base[i].context_good[weather][visibility][rush_hour] / (double)(controller_ID_data_base[i].context_good[weather][visibility][rush_hour] + controller_ID_data_base[i].context_bad[weather][visibility][rush_hour]);

				temp_tv = temp_tv + temp_target_tv;
				temp_count++;
			}
			//if (temp_tv < temp_target_tv) temp_tv = temp_target_tv;
		}
	}
	//decision_thresh = (rand() % 1000) / (double)1000;
	//decision_thresh_I_sharing = temp_tv;

	if(temp_count != 0) decision_thresh_I_sharing = temp_tv / temp_count + 0.05;
	else decision_thresh_I_sharing = 0.5;
	
	decision_thresh_objective = decision_thresh_I_sharing;

	//decision_thresh_I_sharing = threshold;
	//decision_thresh_objective = threshold;

	// make unknown list
	for (i = 0, num_unknown = 0; i < num_ID; i++)
	{
		if (controller_ID_data_base[i].trust_value == NO_HISTORY && (neighbor_a_ID_data_base[i].trust_value != NO_HISTORY || neighbor_b_ID_data_base[i].trust_value != NO_HISTORY))
		{
			num_unknown++;
		}
	}

	list_unknown = (int*)malloc(sizeof(int) * num_unknown);
	list_unknown_check_I_sharing = (int*)malloc(sizeof(int) * num_unknown);

	similarity_list = (double**)malloc(sizeof(double*) * num_unknown);

	for (i = 0, num_unknown = 0; i < num_ID; i++)
	{
		if (controller_ID_data_base[i].trust_value == NO_HISTORY && (neighbor_a_ID_data_base[i].trust_value != NO_HISTORY || neighbor_b_ID_data_base[i].trust_value != NO_HISTORY))
		{
			list_unknown[num_unknown++] = i;
		}
	}

	// allocate memory for simulation
	list_interaction = (int*)malloc(sizeof(int) * NUMBER_INTERACTION_VEHICLE);

	list_I_sharing_trust_value = (double *)malloc(sizeof(double) * NUMBER_INTERACTION_VEHICLE);
	list_objective_trust_value = (double *)malloc(sizeof(double) * NUMBER_INTERACTION_VEHICLE);

	list_behavior_results = (int*)malloc(sizeof(int) * NUMBER_INTERACTION_VEHICLE);
	list_trust_results = (double*)malloc(sizeof(double) * NUMBER_INTERACTION_VEHICLE);

	I_sharing_num_good = (int*)malloc(sizeof(int) * NUMBER_INTERACTION_VEHICLE);
	I_sharing_num_bad = (int*)malloc(sizeof(int) * NUMBER_INTERACTION_VEHICLE);

	objective_num_good = (int*)malloc(sizeof(int) * NUMBER_INTERACTION_VEHICLE);
	objective_num_bad = (int*)malloc(sizeof(int) * NUMBER_INTERACTION_VEHICLE);

	interaction_sim_list = (double**)malloc(sizeof(double*) * NUMBER_INTERACTION_VEHICLE);

	// make I-sharing list
	for (i = 0; i < num_unknown; i++)
	{
		target_ID = list_unknown[i];

		objective_value_a = neighbor_a_ID_data_base[target_ID].trust_value;
		objective_value_b = neighbor_b_ID_data_base[target_ID].trust_value;

		// check existance of target vehicle data in neighbor controller
		if (objective_value_a < 0) reference_flag = 0;
		else reference_flag = 1;

		if (objective_value_b >= 0) reference_flag = reference_flag + 2;

		// leveraging data
		if (reference_flag == 1)
		{
			similarity_list[i] = I_similartiy_estimation(neighbor_a_ID_data_base, target_ID, num_ID);
		}
		else if (reference_flag == 2)
		{
			similarity_list[i] = I_similartiy_estimation(neighbor_b_ID_data_base, target_ID, num_ID);
		}
		else if (reference_flag == 3)
		{
			similarity_a = I_similartiy_estimation(neighbor_a_ID_data_base, target_ID, num_ID);
			similarity_b = I_similartiy_estimation(neighbor_b_ID_data_base, target_ID, num_ID);

			for (j = 0; j < num_ID; j++)
			{
				if (similarity_a[j] < 0)
				{
					if (similarity_b[j] >= 0) similarity_a[j] = similarity_b[j];
				}
				else
				{
					if (similarity_b[j] >= 0) similarity_a[j] = (similarity_a[j] + similarity_b[j]) / 2;
				}
			}

			similarity_list[i] = similarity_a;
			free(similarity_b);
		}

		// estimate initial trust value of target vehicle by using I-sharing method
		temp_tv = estimate_temp_trust(controller_ID_data_base, similarity_list[i], target_ID, weather, visibility, rush_hour, num_ID);

		if (temp_tv < 0) list_unknown_check_I_sharing[i] = 0;
		else list_unknown_check_I_sharing[i] = 1;
	}


	// select vehicles which are iteracting with controller
	num_mal_vehicle = (int)(round(mal_vehicle_portion * NUMBER_INTERACTION_VEHICLE));
	//num_mal_vehicle = (int)(round(MAL_UNKNOWN_PORTION * NUMBER_INTERACTION_VEHICLE));
	num_benign_vehicle = NUMBER_INTERACTION_VEHICLE - num_mal_vehicle;

	for (i = 0, j = 0; i < num_unknown; i++)
	{
		if (list_unknown_check_I_sharing[i] == 1)
		{
			target_ID = list_unknown[i];

			if (controller_ID_data_base[target_ID].status == BENIGN_STATUS)
			{
				if (num_benign_vehicle > 0)
				{
					interaction_sim_list[j] = similarity_list[i];
					list_interaction[j++] = target_ID;
					num_benign_vehicle--;
				}
			}
			else if (controller_ID_data_base[target_ID].status == MALICIOUS_STATUS)
			{
				if (num_mal_vehicle > 0)
				{
					interaction_sim_list[j] = similarity_list[i];
					list_interaction[j++] = target_ID;
					num_mal_vehicle--;
				}
			}
		}
	}

	if (num_benign_vehicle != 0 || num_mal_vehicle != 0)
	{
		// need to handle this case
		printf("ERROR: NO I-SHARING LIST\n");
		return;
	}

	// free unknown list
	free(list_unknown);
	free(list_unknown_check_I_sharing);

	// estimation for simulation
	for (i = 0; i < NUMBER_INTERACTION_VEHICLE; i++)
	{
		target_ID = list_interaction[i];

		objective_value_a = neighbor_a_ID_data_base[target_ID].trust_value;
		objective_value_b = neighbor_b_ID_data_base[target_ID].trust_value;

		// check existance of target vehicle data in neighbor controller
		if (objective_value_a < 0) reference_flag = 0;
		else reference_flag = 1;

		if (objective_value_b >= 0) reference_flag = reference_flag + 2;

		// estimate initial trust value of target vehicle by using reputation
		if (reference_flag == 1)
		{
			objective_value = objective_value_a;
		}
		else if (reference_flag == 2)
		{
			objective_value = objective_value_b;
		}
		else if (reference_flag == 3)
		{
			objective_value = (objective_value_a + objective_value_b) / 2;
		}

		r = rand() % 1000;
		if (r < prob_forge) list_objective_trust_value[i] = 1 - objective_value;
		else list_objective_trust_value[i] = objective_value;

		// estimate initial trust value of target vehicle by using I-sharing method
		sorted_sim_list = similarity_sorting(interaction_sim_list[i], num_ID);
		list_I_sharing_trust_value[i] = estimate_initial_trust(controller_ID_data_base, interaction_sim_list[i], target_ID, weather, visibility, rush_hour, num_ID, sorted_sim_list);

		// use interaction history
		I_sharing_num_good[i] = 0;
		I_sharing_num_bad[i] = 0;

		objective_num_good[i] = 0;
		objective_num_bad[i] = 0;

		free(sorted_sim_list);
	}


	// simulate and check result
	for (j = 0; j < NUMBER_INTERACTION_VEHICLE; j++)
	{
		target_ID = list_interaction[j];
		list_trust_results[j] = 1 - behavior_probability(weather, visibility, rush_hour, &(controller_ID_data_base[target_ID]));
	}

	// select vehicles which are iteracting with controller
	num_mal_vehicle = (int)(round(mal_vehicle_portion * NUMBER_INTERACTION_VEHICLE));
	//num_mal_vehicle = (int)(round(MAL_UNKNOWN_PORTION * NUMBER_INTERACTION_VEHICLE));
	num_benign_vehicle = NUMBER_INTERACTION_VEHICLE - num_mal_vehicle;

	// dynamic threshold
	omega = 76;
	verification_count = 0;

	for (i = 0; i < NUMBER_INTERACTOIN_BEHAVIOR; i++)
	{
		I_TP = 0;
		I_FP = 0;
		I_FN = 0;
		I_TN = 0;

		OB_TP = 0;
		OB_FP = 0;
		OB_FN = 0;
		OB_TN = 0;

		D_I_TP = 0;
		D_I_FP = 0;
		D_I_FN = 0;
		D_I_TN = 0;

		D_OB_TP = 0;
		D_OB_FP = 0;
		D_OB_FN = 0;
		D_OB_TN = 0;

		NONE_true = 0;
		NONE_false = 0;

		if (i > omega)
		{
			verification_count++;

			if (verification_count == 38)
			{
				I_flag = 0;
				ob_flag = 0;

				if (prev_PPV_I_sharing < 0.90) I_flag = I_flag + 1;
				if (prev_PPV_objective < 0.90) ob_flag = ob_flag + 1;

				if (prev_NPV_I_sharing < 0.45) I_flag = I_flag + 2;
				if (prev_NPV_objective < 0.45) ob_flag = ob_flag + 2;

				if(I_flag == 1)
				{
					decision_thresh_I_sharing = decision_thresh_I_sharing + 0.05;
					decision_thresh_objective = decision_thresh_objective + 0.05;
				}
				else if(I_flag == 2)
				{
					decision_thresh_I_sharing = decision_thresh_I_sharing - 0.05;
					decision_thresh_objective = decision_thresh_objective - 0.05;
				}
				else if(I_flag == 3)
				{
					decision_thresh_I_sharing = decision_thresh_I_sharing - 0.05;
					decision_thresh_objective = decision_thresh_objective - 0.05;
				}

				// overflow handling
				if (decision_thresh_I_sharing > 1) decision_thresh_I_sharing = 1;
				if (decision_thresh_I_sharing < 0) decision_thresh_I_sharing = 0;

				if (decision_thresh_objective > 1) decision_thresh_objective = 1;
				if (decision_thresh_objective < 0) decision_thresh_objective = 0;

				verification_count = 0;
			}
		}

		for (j = 0; j < NUMBER_INTERACTION_VEHICLE; j++)
		{
			target_ID = list_interaction[j];
			list_behavior_results[j] = behavior_simulation(weather, visibility, rush_hour, &(controller_ID_data_base[target_ID]));
		}

		for (j = 0; j < NUMBER_INTERACTION_VEHICLE; j++)
		{
			target_ID = list_interaction[j];

			if (i == 0)
			{
				I_sharing_num_good[j] = 1;
				I_sharing_num_bad[j] = 1;

				objective_num_good[j] = 1;
				objective_num_bad[j] = 1;
			}

			I_sharing_direct = I_sharing_num_good[j] / (double)(I_sharing_num_good[j] + I_sharing_num_bad[j]);
			objective_direct = objective_num_good[j] / (double)(objective_num_good[j] + objective_num_bad[j]);

			if (list_I_sharing_trust_value[j] < 0)
			{
				I_sharing_tv = I_sharing_direct;
				printf("ERROR: NO I-shiaring DATA\n");
			}
			else
			{
				I_sharing_tv = (1 - weight) * I_sharing_direct + weight * list_I_sharing_trust_value[j];
			}

			if (list_objective_trust_value[j] < 0)
			{
				objective_tv = objective_direct;
				printf("ERROR: NO I-shiaring DATA\n");
			}
			else
			{
				objective_tv = (1 - weight) * objective_direct + weight * list_objective_trust_value[j];
			}

			// making decision
			if (I_sharing_tv >= decision_thresh_I_sharing) I_sharing_decision = 1;
			else I_sharing_decision = 0;

			if (objective_tv >= decision_thresh_objective) objective_decision = 1;
			else objective_decision = 0;

			r = rand() % 1000;
			if (r < prob_forge)
			{
				if (list_behavior_results[j] == BENIGN_STATUS)
				{
					if (I_sharing_decision == 0) D_I_TN++;
					else D_I_FP++;

					if (objective_decision == 0) D_OB_TN++;
					else D_OB_FP++;

					I_sharing_num_bad[j]++;
					objective_num_bad[j]++;
				}
				else
				{
					if (I_sharing_decision == 0) D_I_FN++;
					else D_I_TP++;

					if (objective_decision == 0) D_OB_FN++;
					else D_OB_TP++;

					I_sharing_num_good[j]++;
					objective_num_good[j]++;
				}
			}
			else
			{
				if (list_behavior_results[j] == BENIGN_STATUS)
				{
					if (I_sharing_decision == 0) D_I_FN++;
					else D_I_TP++;

					if (objective_decision == 0) D_OB_FN++;
					else D_OB_TP++;

					I_sharing_num_good[j]++;
					objective_num_good[j]++;
				}
				else
				{
					if (I_sharing_decision == 0) D_I_TN++;
					else D_I_FP++;

					if (objective_decision == 0) D_OB_TN++;
					else D_OB_FP++;

					I_sharing_num_bad[j]++;
					objective_num_bad[j]++;
				}
			}			

			// check result			
			if (controller_ID_data_base[target_ID].status == BENIGN_STATUS)
			{
				if (I_sharing_decision == 0) I_FN++;
				else I_TP++;

				if (objective_decision == 0) OB_FN++;
				else OB_TP++;

				NONE_true++;
			}
			else
			{
				if (I_sharing_decision == 0) I_TN++;
				else I_FP++;

				if (objective_decision == 0) OB_TN++;
				else OB_FP++;

				NONE_false++;
			}

			fprintf(fp, "%i \t %i \t %i \t %i \t %i \t %i \t %i \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t%i \t %i \n",
				controller_data_base[target_ID].bad,
				controller_data_base[target_ID].good,
				controller_data_base[target_ID].rush_hour,
				controller_data_base[target_ID].visibility,
				controller_data_base[target_ID].weather,
				I_sharing_num_good[target_ID],
				I_sharing_num_bad[target_ID],
				I_sharing_tv,
				decision_thresh_I_sharing,
				probability_base->weather[BAD_WEATHER],
				probability_base->weather[GOOD_WEATHER],
				probability_base->visibility[BAD_VISIBILITY],
				probability_base->visibility[GOOD_VISIBILITY],
				probability_base->rush_hour[RUSH_HOUR],
				probability_base->rush_hour[NOT_RUSH_HOUR],
				I_sharing_decision,
				controller_ID_data_base[target_ID].status);
		}

		if (verification_count == 0)
		{
			// estimate PPV
			if ((D_I_TP + D_I_FP) == 0) prev_PPV_I_sharing = 1;
			else prev_PPV_I_sharing = D_I_TP / (double)(D_I_TP + D_I_FP);

			if ((D_OB_TP + D_OB_FP) == 0) prev_PPV_objective = 1;
			else prev_PPV_objective = D_OB_TP / (double)(D_OB_TP + D_OB_FP);

			// estimate NPV
			if ((D_I_TN + D_I_FN) == 0) prev_NPV_I_sharing = 1;
			else prev_NPV_I_sharing = D_I_TN / (double)(D_I_TN + D_I_FN);

			if ((D_OB_TN + D_OB_FN) == 0) prev_NPV_objective = 1;
			else prev_NPV_objective = D_OB_TN / (double)(D_OB_TN + D_OB_FN);
		}


		// simulation result: F1 score
		//I_recall = I_TP / (double)(I_TP + I_FP);
		//I_precision = I_TP / (double)(I_TP + I_FN);

		//ob_recall = OB_TP / (double)(OB_TP + OB_FP);;
		//ob_precision = OB_TP / (double)(OB_TP + OB_FN);;

		//none_recall = NONE_true / (double)(NONE_true + NONE_false);
		//none_precision = NONE_true / (double)NONE_true;

		//I_trust_result[i] = I_trust_result[i] + (2 * I_recall * I_precision) / (I_recall + I_precision);
		//ob_trust_result[i] = ob_trust_result[i] + (2 * ob_recall * ob_precision) / (ob_recall + ob_precision);
		//none_result[i] = none_result[i] + (2 * none_recall * none_precision) / (none_recall + none_precision);

		// simulation result: accuracy
		I_trust_result[i] = I_trust_result[i] + (I_TP + I_TN) / (double)(I_TP + I_TN + I_FP + I_FN);
		ob_trust_result[i] = ob_trust_result[i] + (OB_TP + OB_TN) / (double)(OB_TP + OB_TN + OB_FP + OB_FN);
		none_result[i] = none_result[i] + NONE_true / (double)(NONE_true + NONE_false);

		// simulation result: balanced accuracy
		//I_trust_result[i] = I_trust_result[i] + ((I_TP / (double)(I_TP + I_FN)) + (I_TN / (double)(I_TN + I_FP))) / 2;
		//ob_trust_result[i] = ob_trust_result[i] + ((OB_TP / (double)(OB_TP + OB_FN)) + (OB_TN / (double)(OB_TN + OB_FP))) / 2;
		//none_result[i] = none_result[i] + (NONE_true / (double)NONE_true) / 2;
	}
	

	// free data allocation
	free(controller_data_base);
	free(controller_ID_data_base);

	free(neighbor_a_data_base);
	free(neighbor_a_ID_data_base);
	free(neighbor_b_data_base);
	free(neighbor_b_ID_data_base);

	free(list_interaction);
	free(list_I_sharing_trust_value);
	free(list_objective_trust_value);

	free(I_sharing_num_good);
	free(I_sharing_num_bad);

	free(objective_num_good);
	free(objective_num_bad);

	free(list_behavior_results);
	free(list_trust_results);

	for (i = 0; i < num_unknown; i++)
	{
		free(similarity_list[i]);
	}
	free(similarity_list);
	free(interaction_sim_list);
}

void simulation_function(int num_ID, int num_controller_ID_data, int num_neighbor_ID_data, int num_data, double weight, double threshold, int prob_forge, double mal_vehicle_portion, FILE *fp)
{
	int i;
	int num_iteration = 250;

	double *I_trust_result;
	double *ob_trust_result;

	double *none_result;

	I_trust_result = (double*)malloc(sizeof(double) * NUMBER_INTERACTOIN_BEHAVIOR);
	ob_trust_result = (double*)malloc(sizeof(double) * NUMBER_INTERACTOIN_BEHAVIOR);

	none_result = (double*)malloc(sizeof(double) * NUMBER_INTERACTOIN_BEHAVIOR);

	for (i = 0; i < NUMBER_INTERACTOIN_BEHAVIOR; i++)
	{
		I_trust_result[i] = 0;
		ob_trust_result[i] = 0;

		none_result[i] = 0;
	}

	// call simulation function
	for (i = 0; i < num_iteration; i++)
	{
		trust_simulation(num_ID, num_controller_ID_data, num_neighbor_ID_data, num_data, weight, threshold, prob_forge, mal_vehicle_portion, fp, I_trust_result, ob_trust_result, none_result);
	}

	// fprintf(fp, "%f\t%f\n", prob_forge / (double)1000, weight);
	// fprintf(fp, "I-sharing\tObjective\tNone\n");
	// for (i = 0; i < NUMBER_INTERACTOIN_BEHAVIOR; i++)
	// {
	// 	fprintf(fp, "%f\t%f\t%f\n", I_trust_result[i] / num_iteration, ob_trust_result[i] / num_iteration, none_result[i] / num_iteration);
	// }
	// fprintf(fp, "\n\n\n");

	// printf("simulation[%f, %f] is done\n", prob_forge / (double)1000, weight);

	free(I_trust_result);
	free(ob_trust_result);

	free(none_result);
}