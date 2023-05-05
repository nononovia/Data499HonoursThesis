import pandas as pd

def get_observations(evidence_df, level):
    evidence_name_list = ['Task done on time=all', 'Task done on time=partial', 'Task done on time=none',
                          'Work accepted by others=all', 'Work accepted by others=partial',
                          'Work accepted by others=none',
                          'Mostly Positive tone', 'Mostly Neutral tone', 'Mostly Negative tone',
                          'Initiate conversations=true', 'Initiate conversations=false',
                          'Help others=deep', 'Help others=shallow', 'Help others=none',
                          'Completes more tasks=true', 'Completes more tasks=false',
                          'Assigns tasks=true', 'Assigns tasks=false',
                          'Review work from others=alot', 'Review work from others=some',
                          'Review work from others=none',
                          'Initiate meeting=true', 'Initiate meeting=false']

    possible_observations_list = ['Task done on time_0', 'Task done on time_1', 'Task done on time_2',
                                  'Work accepted by others_0', 'Work accepted by others_1', 'Work accepted by others_2',
                                  'Positive tone_0', 'Positive tone_1', 'Positive tone_2',
                                  'Initiate conversations_0', 'Initiate conversations_1',
                                  'Help others_0', 'Help others_1', 'Help others_2',
                                  'Completes more tasks_0', 'Completes more tasks_1',
                                  'Assigns tasks_0', 'Assigns tasks_1',
                                  'Review work from others_0', 'Review work from others_1', 'Review work from others_2',
                                  'Initiate meeting_0', 'Initiate meeting_1']

    observable_list = ['Task done on time', 'Work accepted by others', 'Positive tone', 'Initiate conversations',
                       'Help others', 'Completes more tasks', 'Assigns tasks', 'Review work from others',
                       'Initiate meeting']

    result_df = pd.DataFrame(columns=range(0, 13), index=range(0, 501))
    for col in result_df:
        result_df[col] = [[] for _ in range(len(result_df))]

    evidence_df = evidence_df.drop(columns=0)

    for i in range(evidence_df.shape[0]):  # iterate over rows
        for j in range(evidence_df.shape[1]):
            time_slice = j + 1
            for observable in observable_list:
                current_key = (observable, time_slice)
                current_value = evidence_df.iloc[i, j][current_key]
                current_observation = f'{observable}_{current_value}'
                result_df.loc[i, j].append(current_observation)

    result_df.to_csv(f'commitment_observations_{level}.csv')


if __name__ == '__main__':
    evidence_df_low = pd.read_pickle("Commitment_0_observations_example.p")
    evidence_df_mid = pd.read_pickle("Commitment_1_observations_example.p")
    evidence_df_high = pd.read_pickle("Commitment_2_observations_example.p")

    get_observations(evidence_df_low, "low")
    get_observations(evidence_df_mid, "mid")
    get_observations(evidence_df_high, "high")