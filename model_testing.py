import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference
from Commitment_DBN import commitment


def extractDigits(lst):
    return [[el] for el in lst]


def model_testing(model, iterations, node, level):

    # dbn_inf = DBNInference(model)
    #
    # result_high = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    # result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    # result_low = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    evidence_df = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    for i in range(0, iterations):
        print(i)
        sim = model.simulate(1, evidence={(node, 0): level})
        sim_dict = sim.to_dict('records')[0]
        evidence_df.at[i, 0] = sim_dict
        # del sim_dict[(node, 1)]
        # inference_value = dbn_inf.forward_inference([(node, 1)], sim_dict)
        # result = inference_value[(node, 1)].values
        # low_prob = result[0]
        # mid_prob = result[1]
        # high_prob = result[2]
        # result_high.iloc[i, 0] = high_prob
        # result_mid.iloc[i, 0] = mid_prob
        # result_low.iloc[i, 0] = low_prob

        for week in range(1, 14):
            sim = model.simulate(n_samples=1, n_time_slices=week+1)
            sim_dict = sim.to_dict('records')[0]
            evidence_df.at[i, week] = sim_dict
            # del sim_dict[(node, week)]
            # inference_value = dbn_inf.forward_inference([(node, week)], sim_dict)
            # result = inference_value[(node, week)].values
            # low_prob = result[0]
            # mid_prob = result[1]
            # high_prob = result[2]
            # result_high.iloc[i, week] = high_prob
            # result_mid.iloc[i, week] = mid_prob
            # result_low.iloc[i, week] = low_prob

            # result_high.to_pickle(f'{node}_{level}_testing_high.p')
            # result_mid.to_pickle(f'{node}_{level}_testing_mid.p')
            # result_low.to_pickle(f'{node}_{level}_testing_low.p')

    evidence_df.to_csv(f'{node}_{level}_evidence.csv')


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

    observations_df = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    # result_high = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    # result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    # result_low = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    for i in range(0, iterations):
        print(i)
        for week in range(1, 14):
            sim = model.simulate(n_samples=1, n_time_slices=14, evidence={(node, week): level})
            sim_dict = sim.to_dict('records')[0]
            observations_df.at[i, week] = sim_dict
        #     print(f'week{week}')
        #     temp = sim_dict
        #     del temp[(node, week)]
        #     inference_value = dbn_inf.forward_inference([(node, week)], temp)
        #     result = inference_value[(node, week)].values
        #     low_prob = result[0]
        #     mid_prob = result[1]
        #     high_prob = result[2]
        #     result_high.iloc[i, week] = high_prob
        #     result_mid.iloc[i, week] = mid_prob
        #     result_low.iloc[i, week] = low_prob
        # result_high.to_pickle(f'{node}_{level}_testing_high.p')
        # result_mid.to_pickle(f'{node}_{level}_testing_mid.p')
        # result_low.to_pickle(f'{node}_{level}_testing_low.p')
    observations_df.to_pickle(f'{node}_{level}_observations_example.p')


def model_testing4(model, iterations, node, level):
    dbn_inf = DBNInference(model)

    result_high = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_low = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    observations_df = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    for i in range(0, iterations):
        print(i)
        # week 0
        sim = model.simulate(n_samples=1, n_time_slices=1, evidence={(node, 0): level}) #simulating week 0 observations
        sim_dict = sim.to_dict('records')[0]
        observations_df.at[i, 0] = sim_dict

        for week in range(1, 14):
            inference_value = dbn_inf.forward_inference([(node, 1)], sim_dict) # infer current week commitment based on last week observations
            result = inference_value[(node, 1)].values
            new_cpd = TabularCPD((node, 0), variable_card=3, values=extractDigits(result)) # update current commitment cpd
            sim = model.simulate(n_samples=1, n_time_slices=1, virtual_evidence=[new_cpd]) # plug in the new cpd tosimulate current week observations
            sim_dict = sim.to_dict('records')[0]

            observations_df.at[i, week] = sim_dict
            low_prob = result[0]
            mid_prob = result[1]
            high_prob = result[2]
            result_high.iloc[i, week] = high_prob
            result_mid.iloc[i, week] = mid_prob
            result_low.iloc[i, week] = low_prob

    result_high.to_pickle(f'{node}_{level}_testing_high.p')
    result_mid.to_pickle(f'{node}_{level}_testing_mid.p')
    result_low.to_pickle(f'{node}_{level}_testing_low.p')
    observations_df.to_csv(f'{node}_{level}_observations.csv')


def model_testing5(model, iterations, node, level):
    # variation of 3

    dbn_inf = DBNInference(model)

    result_high = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_low = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    observations_df = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    for i in range(0, iterations):
        print(i)
        for week in range(0, 13):
            sim = model.simulate(n_samples=1, n_time_slices=1, evidence={(node, week): level})
            sim_dict = sim.to_dict('records')[0]
            temp = sim_dict
            inference_value = dbn_inf.forward_inference([(node, week+1)], temp)
            result = inference_value[(node, week+1)].values
            observations_df.at[i, week] = sim_dict
            low_prob = result[0]
            mid_prob = result[1]
            high_prob = result[2]
            result_high.iloc[i, week+1] = high_prob
            result_mid.iloc[i, week+1] = mid_prob
            result_low.iloc[i, week+1] = low_prob
            print('')
        result_high.to_pickle(f'{node}_{level}_testing_high.p')
        result_mid.to_pickle(f'{node}_{level}_testing_mid.p')
        result_low.to_pickle(f'{node}_{level}_testing_low.p')
        observations_df.to_csv(f'{node}_{level}_observations.csv')



if __name__ == '__main__':
    commitment_model = commitment()
    model_testing3(commitment_model, 500, "Commitment", 0)
    model_testing3(commitment_model, 500, "Commitment", 1)
    model_testing3(commitment_model, 500, "Commitment", 2)

    # model_testing(commitment_model, 500, 'Commitment', 0)
    # model_testing(commitment_model, 500, 'Commitment', 1)
    # model_testing(commitment_model, 500, 'Commitment', 2)

    # model_testing4(commitment_model, 50, 'Commitment', 0)
    # model_testing4(commitment_model, 50, 'Commitment', 1)
    # model_testing4(commitment_model, 50, 'Commitment', 2)

    # model_testing5(commitment_model, 50, 'Commitment', 0)
    # model_testing5(commitment_model, 50, 'Commitment', 1)
    # model_testing5(commitment_model, 50, 'Commitment', 2)

