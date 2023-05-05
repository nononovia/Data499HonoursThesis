import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference
from Commitment_DBN import commitment
from AgreedUponSystem_DBN import agreed_upon_system_process

def model_testing3(model, iterations, node, level):

    dbn_inf = DBNInference(model)

    # observations_df = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))

    result_high = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    result_low = pd.DataFrame(columns=range(0, 14), index=range(0, iterations))
    for i in range(0, iterations):
        print(i)
        for week in range(1, 14):
            sim = model.simulate(n_samples=1, n_time_slices=14, evidence={(node, week): level})
            sim_dict = sim.to_dict('records')[0]
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
        result_high.to_csv(f'{node}_{level}_testing_high.csv')
        result_mid.to_csv(f'{node}_{level}_testing_mid.csv')
        result_low.to_csv(f'{node}_{level}_testing_low.csv')
    # observations_df.to_pickle(f'{node}_{level}_observations_example.p')

if __name__ == '__main__':

    agreement_model = agreed_upon_system_process()
    model_testing3(agreement_model, 100, "Agreed upon system", 0)
    model_testing3(agreement_model, 100, "Agreed upon system", 1)
    model_testing3(agreement_model, 100, "Agreed upon system", 2)