import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference
from Commitment_DBN import commitment


def model_testing(model, iterations, node, level):

    dbn_inf = DBNInference(model)

    result_high = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_low = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    for i in range(0, iterations):
        print("week 0")
        sim = model.simulate(1, evidence={(node, 0): level})
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[(node, 1)]
        inference_value = dbn_inf.forward_inference([(node, 1)], sim_dict)
        result = inference_value[(node, 1)].values
        low_prob = result[0]
        mid_prob = result[1]
        high_prob = result[2]
        result_high.iloc[i, 0] = high_prob
        result_mid.iloc[i, 0] = mid_prob
        result_low.iloc[i, 0] = low_prob

        for week in range(1, 14):
            print(f'week {week}')
            sim = model.simulate(n_samples=1, n_time_slices=week+1)
            sim_dict = sim.to_dict('records')[0]
            del sim_dict[(node, week)]
            inference_value = dbn_inf.forward_inference([(node, week)], sim_dict)
            result = inference_value[(node, week)].values
            low_prob = result[0]
            mid_prob = result[1]
            high_prob = result[2]
            result_high.iloc[i, week] = high_prob
            result_mid.iloc[i, week] = mid_prob
            result_low.iloc[i, week] = low_prob

            result_high.to_pickle(f'{node}_{level}_testing_high.p')
            result_mid.to_pickle(f'{node}_{level}_testing_mid.p')
            result_low.to_pickle(f'{node}_{level}_testing_low.p')


def model_testing2(model, iterations, node, level):

    dbn_inf = DBNInference(model)

    result_high = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_low = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    for i in range(0, iterations):
        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={(node, 0): level})
        sim_dict = sim.to_dict('records')[0]
        for week in range(0, 13):
            print(f'week{week}')
            temp = sim_dict
            del temp[(node, week+1)]
            inference_value = dbn_inf.forward_inference([(node, week+1)], temp)
            result = inference_value[(node, week+1)].values
            low_prob = result[0]
            mid_prob = result[1]
            high_prob = result[2]
            result_high.iloc[i, week] = high_prob
            result_mid.iloc[i, week] = mid_prob
            result_low.iloc[i, week] = low_prob
        result_high.to_pickle(f'{node}_{level}_testing_high.p')
        result_mid.to_pickle(f'{node}_{level}_testing_mid.p')
        result_low.to_pickle(f'{node}_{level}_testing_low.p')


def model_testing3(model, iterations, node, level):

    dbn_inf = DBNInference(model)

    result_high = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_low = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    for i in range(0, iterations):
        for week in range(1, 14):
            sim = model.simulate(n_samples=1, n_time_slices=14, evidence={(node, week): level})
            sim_dict = sim.to_dict('records')[0]
            print(f'week{week}')
            temp = sim_dict
            del temp[(node, week)]
            inference_value = dbn_inf.forward_inference([(node, week)], temp)
            result = inference_value[(node, week)].values
            low_prob = result[0]
            mid_prob = result[1]
            high_prob = result[2]
            result_high.iloc[i, week] = high_prob
            result_mid.iloc[i, week] = mid_prob
            result_low.iloc[i, week] = low_prob
        result_high.to_pickle(f'{node}_{level}_testing_high.p')
        result_mid.to_pickle(f'{node}_{level}_testing_mid.p')
        result_low.to_pickle(f'{node}_{level}_testing_low.p')


if __name__ == '__main__':
    commitment_model = commitment()
    model_testing3(commitment_model, 500, "Commitment", 0)
    model_testing3(commitment_model, 500, "Commitment", 1)
    model_testing3(commitment_model, 500, "Commitment", 2)
