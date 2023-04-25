import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

    d = {"week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentA": student_A_result_average,
         "studentB": student_B_result_average,
         "studentC": student_C_result_average,
         "studentD": student_D_result_average,
         "studentF": student_F_result_average,
    }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='week', y='studentA', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentB', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentC', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentD', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentF', ax=ax, x_compat=True)
    plt.xticks(df['week'])
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

    d = {"week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentA": student_A_result_average,
         "studentA1": student_A2_result_average,
    }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='week', y='studentA', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentA1', ax=ax, x_compat=True)
    plt.xticks(df['week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
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

    d = {"week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentB": student_B_result_average,
         "studentB1": student_B1_result_average,
    }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='week', y='studentB', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentB1', ax=ax, x_compat=True)
    plt.xticks(df['week'])
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

    d = {"week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentA": student_A_result_average,
         "studentB": student_B_result_average,
         "studentC": student_C_result_average,
         "studentD": student_D_result_average,
         "studentF": student_F_result_average,
    }
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='week', y='studentA', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentB', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentC', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentD', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentF', ax=ax, x_compat=True)
    plt.xticks(df['week'])
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

    d = {"week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentA": student_A_result_average,
         "studentA1": student_A2_result_average,
    }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='week', y='studentA', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentA1', ax=ax, x_compat=True)
    plt.xticks(df['week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.ylim(0, 1)
    plt.title("Strong inclination")
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

    d = {"week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         "studentB": student_B_result_average,
         "studentB1": student_B1_result_average,
    }
    # "student3": student_3_result}
    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='week', y='studentB', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y='studentB1', ax=ax, x_compat=True)
    plt.xticks(df['week'])
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

    d = {"week": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         f'{node}_low': low_result_average,
         f'{node}_mid': mid_result_average,
         f'{node}_high': high_result_average,
         }

    df = pd.DataFrame(d)
    ax = plt.gca()
    df.plot(kind='line', x='week', y=f'{node}_low', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y= f'{node}_mid', ax=ax, x_compat=True)
    df.plot(kind='line', x='week', y= f'{node}_high', ax=ax, x_compat=True)

    plt.xticks(df['week'])
    plt.ylabel(f'Probability of them being a {level} commitment student')
    plt.title("Testing Result")
    plt.ylim(0, 1)
    plt.show()
    plt.close()


# def graphing(level, *file_names):
#     for file in file_names:
#         student_result = pd.read_pickle(f'{file}_{level}.p')
#         student_result_average = student_result.apply(np.mean, axis="rows")

if __name__ == '__main__':
    # graphing_test_result("high", "Commitment")
    # graphing_test_result("mid", "Commitment")
    # graphing_test_result("low", "Commitment")

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
    graphing_As_strong_impression("high")
    graphing_As_strong_impression("mid")
    graphing_As_strong_impression("low")