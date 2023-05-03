import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


def graphing_all(level):
    student_A_result = pd.read_pickle(f'studentA_{level}.p')
    student_A_result_average = student_A_result.apply(np.mean, axis="rows")

    student_B_result = pd.read_pickle(f'studentB1_{level}.p')
    student_B_result_average = student_B_result.apply(np.mean, axis="rows")

    student_C_result = pd.read_pickle(f'studentC_{level}.p')
    student_C_result_average = student_C_result.apply(np.mean, axis="rows")

    student_D_result = pd.read_pickle(f'studentD_{level}.p')
    student_D_result_average = student_D_result.apply(np.mean, axis="rows")

    student_F_result = pd.read_pickle(f'studentF_{level}.p')
    student_F_result_average = student_F_result.apply(np.mean, axis="rows")

    if level == "high":
        student_A_result_average[0] = 0.2
    elif level == "mid":
        student_A_result_average[0] = 0.6
    else:
        student_A_result_average[0] = 0.2

    if level == "high":
        student_B_result_average[0] = 0.2
    elif level == "mid":
        student_B_result_average[0] = 0.6
    else:
        student_B_result_average[0] = 0.2

    if level == "high":
        student_C_result_average[0] = 0.2
    elif level == "mid":
        student_C_result_average[0] = 0.6
    else:
        student_C_result_average[0] = 0.2

    if level == "high":
        student_D_result_average[0] = 0.2
    elif level == "mid":
        student_D_result_average[0] = 0.6
    else:
        student_D_result_average[0] = 0.2

    if level == "high":
        student_F_result_average[0] = 0.2
    elif level == "mid":
        student_F_result_average[0] = 0.6
    else:
        student_F_result_average[0] = 0.2

    d = {"Week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "StudentA": student_A_result_average,
         "StudentB": student_B_result_average,
         "StudentC": student_C_result_average,
         "StudentD": student_D_result_average,
         "StudentF": student_F_result_average,
         }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='Week', y='StudentA', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='StudentB', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='StudentC', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='StudentD', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='StudentF', ax=ax, x_compat=True)
    plt.xticks(df['Week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.ylim(0, 1)
    plt.show()
    plt.close()


def graphing_As(level):
    student_A_result = pd.read_pickle(f'studentA_{level}.p')
    student_A_result_average = student_A_result.apply(np.mean, axis="rows")

    student_A2_result = pd.read_pickle(f'studentA2_{level}.p')
    student_A2_result_average = student_A2_result.apply(np.mean, axis="rows")

    if level == "high":
        student_A_result_average[0] = 0.2
    elif level == "mid":
        student_A_result_average[0] = 0.6
    else:
        student_A_result_average[0] = 0.2

    if level == "high":
        student_A2_result_average[0] = 0.2
    elif level == "mid":
        student_A2_result_average[0] = 0.6
    else:
        student_A2_result_average[0] = 0.2

    d = {"Week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "StudentA": student_A_result_average,
         "StudentA1": student_A2_result_average,
         }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='Week', y='StudentA', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='StudentA1', ax=ax, x_compat=True)
    plt.xticks(df['Week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.title("Easily Impressionable")
    plt.ylim(0, 1)
    plt.show()
    plt.close()


def graphing_Bs(level):
    student_B_result = pd.read_pickle(f'studentB_{level}.p')
    student_B_result_average = student_B_result.apply(np.mean, axis="rows")

    student_B1_result = pd.read_pickle(f'studentB1_{level}.p')
    student_B1_result_average = student_B1_result.apply(np.mean, axis="rows")

    if level == "high":
        student_B_result_average[0] = 0.2
    elif level == "mid":
        student_B_result_average[0] = 0.6
    else:
        student_B_result_average[0] = 0.2

    if level == "high":
        student_B1_result_average[0] = 0.2
    elif level == "mid":
        student_B1_result_average[0] = 0.6
    else:
        student_B1_result_average[0] = 0.2

    d = {"Week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentB": student_B_result_average,
         "studentB1": student_B1_result_average,
         }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='Week', y='studentB', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='studentB1', ax=ax, x_compat=True)
    plt.xticks(df['Week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.ylim(0, 1)
    plt.show()
    plt.close()


def graphing_all_strong_impression(level):
    student_A_result = pd.read_pickle(f'studentA_strong_impression_{level}.p')
    student_A_result_average = student_A_result.apply(np.mean, axis="rows")

    student_B_result = pd.read_pickle(f'studentB1_strong_impression_{level}.p')
    student_B_result_average = student_B_result.apply(np.mean, axis="rows")

    student_C_result = pd.read_pickle(f'studentC_strong_impression_{level}.p')
    student_C_result_average = student_C_result.apply(np.mean, axis="rows")

    student_D_result = pd.read_pickle(f'studentD_strong_impression_{level}.p')
    student_D_result_average = student_D_result.apply(np.mean, axis="rows")

    student_F_result = pd.read_pickle(f'studentF_strong_impression_{level}.p')
    student_F_result_average = student_F_result.apply(np.mean, axis="rows")

    if level == "high":
        student_A_result_average[0] = 0.2
    elif level == "mid":
        student_A_result_average[0] = 0.6
    else:
        student_A_result_average[0] = 0.2

    if level == "high":
        student_B_result_average[0] = 0.2
    elif level == "mid":
        student_B_result_average[0] = 0.6
    else:
        student_B_result_average[0] = 0.2

    if level == "high":
        student_C_result_average[0] = 0.2
    elif level == "mid":
        student_C_result_average[0] = 0.6
    else:
        student_C_result_average[0] = 0.2

    if level == "high":
        student_D_result_average[0] = 0.2
    elif level == "mid":
        student_D_result_average[0] = 0.6
    else:
        student_D_result_average[0] = 0.2

    if level == "high":
        student_F_result_average[0] = 0.2
    elif level == "mid":
        student_F_result_average[0] = 0.6
    else:
        student_F_result_average[0] = 0.2

    d = {"Week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentA": student_A_result_average,
         "studentB": student_B_result_average,
         "studentC": student_C_result_average,
         "studentD": student_D_result_average,
         "studentF": student_F_result_average,
         }
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='Week', y='studentA', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='studentB', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='studentC', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='studentD', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='studentF', ax=ax, x_compat=True)
    plt.xticks(df['Week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.ylim(0, 1)
    plt.title("Strong inclination")
    plt.show()
    plt.close()


def graphing_As_strong_impression(level):
    student_A_result = pd.read_pickle(f'studentA_strong_impression_{level}.p')
    student_A_result_average = student_A_result.apply(np.mean, axis="rows")

    student_A2_result = pd.read_pickle(f'studentA2_strong_impression_{level}.p')
    student_A2_result_average = student_A2_result.apply(np.mean, axis="rows")

    if level == "high":
        student_A_result_average[0] = 0.2
    elif level == "mid":
        student_A_result_average[0] = 0.6
    else:
        student_A_result_average[0] = 0.2

    if level == "high":
        student_A2_result_average[0] = 0.2
    elif level == "mid":
        student_A2_result_average[0] = 0.6
    else:
        student_A2_result_average[0] = 0.2

    d = {"Week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "StudentA": student_A_result_average,
         "StudentA1": student_A2_result_average,
         }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='Week', y='StudentA', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='StudentA1', ax=ax, x_compat=True)
    plt.xticks(df['Week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.ylim(0, 1)
    plt.title("Hard to Impress")
    plt.show()
    plt.close()


def graphing_Bs_strong_impression(level):
    student_B_result = pd.read_pickle(f'studentB_strong_impression_{level}.p')
    student_B_result_average = student_B_result.apply(np.mean, axis="rows")

    student_B1_result = pd.read_pickle(f'studentB1_strong_impression_{level}.p')
    student_B1_result_average = student_B1_result.apply(np.mean, axis="rows")

    if level == "high":
        student_B_result_average[0] = 0.2
    elif level == "mid":
        student_B_result_average[0] = 0.6
    else:
        student_B_result_average[0] = 0.2

    if level == "high":
        student_B1_result_average[0] = 0.2
    elif level == "mid":
        student_B1_result_average[0] = 0.6
    else:
        student_B1_result_average[0] = 0.2

    d = {"Week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentB": student_B_result_average,
         "studentB1": student_B1_result_average,
         }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='Week', y='studentB', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y='studentB1', ax=ax, x_compat=True)
    plt.xticks(df['Week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.title("Strong inclination")
    plt.ylim(0, 1)
    plt.show()
    plt.close()


def graphing_test_result(level, node):
    low_result = pd.read_pickle(f'{node}_0_testing_{level}.p')
    low_result_average = low_result.apply(np.mean, axis="rows")

    mid_result = pd.read_pickle(f'{node}_1_testing_{level}.p')
    mid_result_average = mid_result.apply(np.mean, axis="rows")

    high_result = pd.read_pickle(f'{node}_2_testing_{level}.p')
    high_result_average = high_result.apply(np.mean, axis="rows")

    # low_result.to_csv("Agreed_upon_process_low_1.csv")
    # mid_result.to_csv("Agreed_upon_process_mid_1.csv")
    # high_result.to_csv("Agreed_upon_process_high_1.csv")

    d = {"Week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    # d = {"Month": [0, 1, 2, 3],
         f'{node}_low': low_result_average,
         f'{node}_mid': mid_result_average,
         f'{node}_high': high_result_average,
         }

    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='Week', y=f'{node}_low', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y=f'{node}_mid', ax=ax, x_compat=True)
    df.plot(kind='line', x='Week', y=f'{node}_high', ax=ax, x_compat=True)

    plt.xticks(df['Week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.title("Testing Result")
    plt.ylim(0, 1)
    plt.show()
    plt.close()


def histograms(evidence_df, level):
    """
    {('Commitment', 0): 0, ('Mutual accountability', 0): 0, ('Trust', 0): 1, ('Completes the work', 0): 1, ('Enthusiastic', 0): 1, ('Go above and beyond', 0): 1, ('Takes charge', 0): 1, ('Commitment', 1): 0, ('Mutual accountability', 1): 0, ('Trust', 1): 2, ('Completes the work', 1): 0, ('Enthusiastic', 1): 0, ('Go above and beyond', 1): 1, ('Takes charge', 1): 1, ('Task done on time', 0): 1, ('Work accepted by others', 0): 1, ('Task done on time', 1): 1, ('Work accepted by others', 1): 2, ('Positive tone', 0): 1, ('Initiate conversations', 0): 1, ('Positive tone', 1): 2, ('Initiate conversations', 1): 1, ('Help others', 0): 2, ('Completes more tasks', 0): 1, ('Help others', 1): 2, ('Completes more tasks', 1): 1, ('Assigns tasks', 0): 1, ('Review work from others', 0): 2, ('Initiate meeting', 0): 1, ('Assigns tasks', 1): 1, ('Review work from others', 1): 2, ('Initiate meeting', 1): 1}

    """

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

    result_df = pd.DataFrame(columns=possible_observations_list)
    result_df.loc[len(result_df)] = 0
    evidence_df = evidence_df.drop(columns=0)

    for i in range(evidence_df.shape[0]):  # iterate over rows
        for j in range(evidence_df.shape[1]):
            time_slice = j + 1
            for observable in observable_list:
                current_key = (observable, time_slice)
                current_value = evidence_df.iloc[i, j][current_key]
                current_observation = f'{observable}_{current_value}'
                result_df.loc[0, current_observation] += 1

    # plot bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(evidence_name_list, result_df.iloc[0], color='blue')
    # set font size and rotate x labels
    plt.xticks(rotation=90, ha='right', fontsize=11)
    # adjust bottom margin
    plt.subplots_adjust(bottom=0.43)
    plt.title(f'Distribution of generated observations for {level} level commitment students')
    plt.show()
    # loop through all keys that contain "task done on time" then assign the levels accordingly


# def graphing(level, *file_names):
#     for file in file_names:
#         student_result = pd.read_pickle(f'{file}_{level}.p')
#         student_result_average = student_result.apply(np.mean, axis="rows")

def graphing_all_groups(level):
    group_A_result = pd.read_pickle(f'Group_A_{level}.p')
    group_A_result_average = group_A_result.apply(np.mean, axis="rows")

    group_B_result1 = pd.read_pickle(f'Group_B_{level}.p')
    group_B_result2 = pd.read_pickle(f'Group_B_{level}_2.p')
    group_B_result = pd.concat([group_B_result1, group_B_result2])
    group_B_result_average = group_B_result.apply(np.mean, axis="rows")

    group_C_result1 = pd.read_pickle(f'Group_C_{level}.p')
    group_C_result2 = pd.read_pickle(f'Group_C_{level}_2.p')
    group_C_result = pd.concat([group_C_result1, group_C_result2])
    group_C_result_average = group_C_result.apply(np.mean, axis="rows")

    print(len(group_C_result))

    d = {"month": [0, 1, 2, 3],
         "groupA": group_A_result_average,
         "groupB": group_B_result_average,
         "groupC": group_C_result_average,
         }
    # "group3": group_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='month', y='groupA', ax=ax, x_compat=True)
    df.plot(kind='line', x='month', y='groupB', ax=ax, x_compat=True)
    df.plot(kind='line', x='month', y='groupC', ax=ax, x_compat=True)
    plt.xticks(df['month'])
    plt.ylabel(f'Probability of the group have {level} level agreed upon process')
    plt.ylim(0, 1)
    plt.show()
    plt.close()


def histograms_group(evidence_df, level):
    """
    {('Commitment', 0): 0, ('Mutual accountability', 0): 0, ('Trust', 0): 1, ('Completes the work', 0): 1, ('Enthusiastic', 0): 1, ('Go above and beyond', 0): 1, ('Takes charge', 0): 1, ('Commitment', 1): 0, ('Mutual accountability', 1): 0, ('Trust', 1): 2, ('Completes the work', 1): 0, ('Enthusiastic', 1): 0, ('Go above and beyond', 1): 1, ('Takes charge', 1): 1, ('Task done on time', 0): 1, ('Work accepted by others', 0): 1, ('Task done on time', 1): 1, ('Work accepted by others', 1): 2, ('Positive tone', 0): 1, ('Initiate conversations', 0): 1, ('Positive tone', 1): 2, ('Initiate conversations', 1): 1, ('Help others', 0): 2, ('Completes more tasks', 0): 1, ('Help others', 1): 2, ('Completes more tasks', 1): 1, ('Assigns tasks', 0): 1, ('Review work from others', 0): 2, ('Initiate meeting', 0): 1, ('Assigns tasks', 1): 1, ('Review work from others', 1): 2, ('Initiate meeting', 1): 1}

    """

    evidence_name_list = ["generally on track in completing the project=true",
                          "generally on track in completing the project=false",
                          "members doing tasks associated with role=all",
                          "members doing tasks associated with role=some",
                          "members doing tasks associated with role=none",
                          "task assigned based on role=all", "task assigned based on role=some",
                          "task assigned based on role-none",
                          "reviewer reviews corresponding task=all", "reviewer reviews corresponding task=some",
                          "reviewer reviews corresponding task=none",
                          "reviewer provides meaningful feedback=all", "reviewer provides meaningful feedback=some",
                          "reviewer provides meaningful feedback=none",
                          "reviewer reviews in a timely manner=true", "reviewer reviews in a timely manner=false",
                          "avg number of points on each task=high", "avg number of points on each task=mid",
                          "avg number of points on each task=low",
                          "num words in task description=high", "num words in task description=mid",
                          "num words in task description=low",
                          "task being rejected multiple times=true", "task being rejected multiple times=true",
                          "duration of task in progress state=long", "duration of task in progress state=mid",
                          "duration of task in progress state=short",
                          "task assigned based on timeline=true", "task assigned based on timeline=false",
                          "completion of task follows timeline=true", "completion of task follows timeline=false"
                          ]

    evidence_name_list = list(range(0, 31))

    possible_observations_list = ["generally on track_0", "generally on track_1",
                                  "member following role_0", "member following role_1", "member following role_2",
                                  "task assigned based on role_0", "task assigned based on role_1",
                                  "task assigned based on role_2",
                                  "reviewer reviews task_0", "reviewer reviews task_1", "reviewer reviews task_2",
                                  "reviewer provides feedback_0", "reviewer provides feedback_1",
                                  "reviewer provides feedback_2",
                                  "reviewer reviews timely_0", "reviewer reviews timely_1",
                                  "avg number of points_0", "avg number of points_1", "avg number of points_2",
                                  "num words in task description_0", "num words in task description_1",
                                  "num words in task description_2",
                                  "task being rejected_0", "task being rejected_1",
                                  "duration of task in progress_0", "duration of task in progress_1",
                                  "duration of task in progress_2",
                                  "task assigned based on timeline_0", "task assigned based on timeline_1",
                                  "completion time follows timeline_0", "completion time follows timeline_1"
                                  ]

    observable_list = ["generally on track", "member following role",
                       "task assigned based on role",
                       "reviewer reviews task",
                       "reviewer provides feedback",
                       "reviewer reviews timely",
                       "avg number of points",
                       "num words in task description",
                       "task being rejected",
                       "duration of task in progress",
                       "task assigned based on timeline",
                       "completion time follows timeline"]

    result_df = pd.DataFrame(columns=possible_observations_list)
    result_df.loc[len(result_df)] = 0
    evidence_df = evidence_df.drop(columns=0)

    for i in range(evidence_df.shape[0]):  # iterate over rows
        for j in range(evidence_df.shape[1]):
            time_slice = j + 1
            for observable in observable_list:
                current_key = (observable, time_slice)
                current_value = evidence_df.iloc[i, j][current_key]
                current_observation = f'{observable}_{current_value}'
                result_df.loc[0, current_observation] += 1

    # plot bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(evidence_name_list, result_df.iloc[0], color='blue')
    # set font size and rotate x labels
    # plt.xticks(rotation=50, ha='right', fontsize=11)
    # adjust bottom margin
    # plt.subplots_adjust(bottom=0.55, left=0.3)
    plt.title(f'Distribution of observations for groups with {level} level for agreed upon system')
    plt.show()
    # loop through all keys that contain "task done on time" then assign the levels accordingly


if __name__ == '__main__':
    # graphing_test_result("high", "Commitment")
    # graphing_test_result("mid", "Commitment")
    # graphing_test_result("low", "Commitment")

    # evidence_df_low = pd.read_pickle("Commitment_0_observations_example.p")
    # evidence_df_mid = pd.read_pickle("Commitment_1_observations_example.p")
    # evidence_df_high = pd.read_pickle("Commitment_2_observations_example.p")
    # histograms(evidence_df_low, "low")
    # histograms(evidence_df_mid, "mid")
    # histograms(evidence_df_high, "high")

    # graphing_all("high")
    # graphing_all("mid")
    # graphing_all("low")
    #
    # graphing_Bs("high")
    # graphing_Bs("mid")
    # graphing_Bs("low")
    #
    # graphing_As("high")
    # graphing_As("mid")
    # graphing_As("low")
    #
    # graphing_all_strong_impression("high")
    # graphing_all_strong_impression("mid")
    # graphing_all_strong_impression("low")
    #
    # graphing_Bs_strong_impression("high")
    # graphing_Bs_strong_impression("mid")
    # graphing_Bs_strong_impression("low")
    #
    # graphing_As_strong_impression("high")
    # graphing_As_strong_impression("mid")
    # graphing_As_strong_impression("low")

    # graphing_test_result("high", "Agreed upon system")
    # graphing_test_result("mid", "Agreed upon system")
    # graphing_test_result("low", "Agreed upon system")

    # graphing_all_groups("high")
    # graphing_all_groups("mid")
    # graphing_all_groups("low")

    evidence_df_low = pd.read_pickle("Agreed upon system_0_observations_example.p")
    evidence_df_mid = pd.read_pickle("Agreed upon system_1_observations_example.p")
    evidence_df_high = pd.read_pickle("Agreed upon system_2_observations_example.p")

    histograms_group(evidence_df_high, "high")
    histograms_group(evidence_df_mid, "mid")
    histograms_group(evidence_df_low, "low")
