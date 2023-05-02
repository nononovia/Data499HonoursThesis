import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference
import numpy as np
from matplotlib import pyplot as plt

mutual_accountability_node_0 = ('Mutual accountability', 0)
trust_node_0 = ('Trust', 0)
commitment_node_0 = ('Commitment', 0)
completes_the_work_node_0 = ('Completes the work', 0)
enthusiastic_node_0 = ('Enthusiastic', 0)
go_above_and_beyond_node_0 = ('Go above and beyond', 0)
takes_charge_node_0 = ('Takes charge', 0)
task_done_on_time_node_0 = ('Task done on time', 0)
work_accepted_by_others_node_0 = ('Work accepted by others', 0)
positive_tone_node_0 = ('Positive tone', 0)
initiate_conversations_node_0 = ('Initiate conversations', 0)
help_others_node_0 = ('Help others', 0)
completes_more_tasks_node_0 = ('Completes more tasks', 0)
assigns_tasks_node_0 = ('Assigns tasks', 0)
review_work_from_others_node_0 = ('Review work from others', 0)
initiate_meeting_node_0 = ('Initiate meeting', 0)

mutual_accountability_node_1 = ('Mutual accountability', 1)
trust_node_1 = ('Trust', 1)
commitment_node_1 = ('Commitment', 1)
completes_the_work_node_1 = ('Completes the work', 1)
enthusiastic_node_1 = ('Enthusiastic', 1)
go_above_and_beyond_node_1 = ('Go above and beyond', 1)
takes_charge_node_1 = ('Takes charge', 1)
task_done_on_time_node_1 = ('Task done on time', 1)
work_accepted_by_others_node_1 = ('Work accepted by others', 1)
positive_tone_node_1 = ('Positive tone', 1)
initiate_conversations_node_1 = ('Initiate conversations', 1)
help_others_node_1 = ('Help others', 1)
completes_more_tasks_node_1 = ('Completes more tasks', 1)
assigns_tasks_node_1 = ('Assigns tasks', 1)
review_work_from_others_node_1 = ('Review work from others', 1)
initiate_meeting_node_1 = ('Initiate meeting', 1)

MAX_ITERATIONS = 100


def commitment():
    model = DynamicBayesianNetwork([
        (commitment_node_0, mutual_accountability_node_0),
        (commitment_node_0, trust_node_0),
        (commitment_node_0, completes_the_work_node_0),
        (commitment_node_0, enthusiastic_node_0),
        (commitment_node_0, go_above_and_beyond_node_0),
        (commitment_node_0, takes_charge_node_0),
        (completes_the_work_node_0, task_done_on_time_node_0),
        (completes_the_work_node_0, work_accepted_by_others_node_0),
        (enthusiastic_node_0, positive_tone_node_0),
        (enthusiastic_node_0, initiate_conversations_node_0),
        (go_above_and_beyond_node_0, help_others_node_0),
        (go_above_and_beyond_node_0, completes_more_tasks_node_0),
        (takes_charge_node_0, assigns_tasks_node_0),
        (takes_charge_node_0, review_work_from_others_node_0),
        (takes_charge_node_0, initiate_meeting_node_0),
        (commitment_node_0, commitment_node_1),
        (commitment_node_1, mutual_accountability_node_1),
        (commitment_node_1, trust_node_1),
        (commitment_node_1, completes_the_work_node_1),
        (commitment_node_1, enthusiastic_node_1),
        (commitment_node_1, go_above_and_beyond_node_1),
        (commitment_node_1, takes_charge_node_1),
        (completes_the_work_node_1, task_done_on_time_node_1),
        (completes_the_work_node_1, work_accepted_by_others_node_1),
        (enthusiastic_node_1, positive_tone_node_1),
        (enthusiastic_node_1, initiate_conversations_node_1),
        (go_above_and_beyond_node_1, help_others_node_1),
        (go_above_and_beyond_node_1, completes_more_tasks_node_1),
        (takes_charge_node_1, assigns_tasks_node_1),
        (takes_charge_node_1, review_work_from_others_node_1),
        (takes_charge_node_1, initiate_meeting_node_1)
    ])

    # Defining individual CPDs.

    # T-1
    cpd_commitment = TabularCPD(commitment_node_0, variable_card=3, values=[
        [0.2],  # low
        [0.6],  # mid
        [0.2]  # high
    ])

    # variable card = num rows, evidence card = num_cols

    cpd_mutual_accountability = TabularCPD(variable=mutual_accountability_node_0,
                                           variable_card=3,
                                           values=[[0.9, 0.3, 0.1],
                                                   [0.09, 0.5, 0.3],
                                                   [0.01, 0.2, 0.6]],
                                           evidence=[commitment_node_0],
                                           evidence_card=[3])

    cpd_trust = TabularCPD(variable=trust_node_0, variable_card=3,
                           values=[[0.4, 0.3, 0.1],
                                   [0.3, 0.4, 0.3],
                                   [0.3, 0.3, 0.6]],
                           evidence=[commitment_node_0],
                           evidence_card=[3])

    cpd_completes_work = TabularCPD(variable=completes_the_work_node_0, variable_card=3,
                                    values=[[0.03, 0.3, 0.8],
                                            [0.5, 0.6, 0.15],
                                            [0.47, 0.1, 0.05]],
                                    evidence=[commitment_node_0],
                                    evidence_card=[3])

    cpd_enthusiastic = TabularCPD(variable=enthusiastic_node_0, variable_card=3,
                                  values=[[0.6, 0.3, 0.2],
                                          [0.3, 0.5, 0.3],
                                          [0.1, 0.2, 0.5]],
                                  evidence=[commitment_node_0],
                                  evidence_card=[3])

    cpd_go_above_beyond = TabularCPD(variable=go_above_and_beyond_node_0, variable_card=2,
                                     values=[[0.01, 0.1, 0.45],
                                             [0.99, 0.9, 0.55]],
                                     evidence=[commitment_node_0],
                                     evidence_card=[3])

    cpd_takes_charge = TabularCPD(variable=takes_charge_node_0, variable_card=2,
                                  values=[[0.01, 0.2, 0.9],
                                          [0.99, 0.8, 0.1]],
                                  evidence=[commitment_node_0],
                                  evidence_card=[3])

    cpd_task_done_on_time = TabularCPD(variable=task_done_on_time_node_0, variable_card=3,
                                       values=[[0.9, 0, 0],
                                               [0.09, 0.95, 0],
                                               [0.01, 0.05, 1]],
                                       evidence=[completes_the_work_node_0],
                                       evidence_card=[3])

    cpd_work_accepted_by_others = TabularCPD(variable=work_accepted_by_others_node_0, variable_card=3,
                                             values=[[0.8, 0, 0],
                                                     [0.15, 0.9, 0],
                                                     [0.05, 0.1, 1]],
                                             evidence=[completes_the_work_node_0],
                                             evidence_card=[3])

    cpd_positive_tone = TabularCPD(variable=positive_tone_node_0, variable_card=3,
                                   values=[[0.1, 0.1, 0.6],
                                           [0.5, 0.8, 0.35],
                                           [0.4, 0.1, 0.05]],
                                   evidence=[enthusiastic_node_0],
                                   evidence_card=[3])

    cpd_initiate_conversation = TabularCPD(variable=initiate_conversations_node_0, variable_card=2,
                                           values=[[0.05, 0.2, 0.7],
                                                   [0.95, 0.8, 0.3]],
                                           evidence=[enthusiastic_node_0],
                                           evidence_card=[3])

    cpd_help_others = TabularCPD(variable=help_others_node_0, variable_card=3,
                                 values=[[0.5, 0.1],
                                         [0.3, 0.2],
                                         [0.2, 0.7]],
                                 evidence=[go_above_and_beyond_node_0],
                                 evidence_card=[2])

    cpd_complete_more_tasks = TabularCPD(variable=completes_more_tasks_node_0, variable_card=2,
                                         values=[[0.8, 0.01],
                                                 [0.2, 0.99]],
                                         evidence=[go_above_and_beyond_node_0],
                                         evidence_card=[2])

    cpd_assigns_tasks = TabularCPD(variable=assigns_tasks_node_0, variable_card=2,
                                   values=[[0.8, 0.1],
                                           [0.2, 0.9]],
                                   evidence=[takes_charge_node_0],
                                   evidence_card=[2])

    cpd_initiate_meetings = TabularCPD(variable=initiate_meeting_node_0, variable_card=2,
                                       values=[[0.85, 0.05],
                                               [0.15, 0.95]],
                                       evidence=[takes_charge_node_0],
                                       evidence_card=[2])

    cpd_review_others_work = TabularCPD(variable=review_work_from_others_node_0, variable_card=3,
                                        values=[[0.3, 0.1],
                                                [0.6, 0.5],
                                                [0.1, 0.4]],
                                        evidence=[takes_charge_node_0],
                                        evidence_card=[2])

    # Transition
    commitment_transitional_cpd = TabularCPD(variable=commitment_node_1, variable_card=3,
                                             values=[[0.8, 0.15, 0.05],
                                                     [0.15, 0.8, 0.15],
                                                     [0.05, 0.05, 0.8]],
                                             evidence=[commitment_node_0],
                                             evidence_card=[3])

    # commitment_transitional_cpd = TabularCPD(variable=commitment_node_1, variable_card=3,
    #                                          values=[[0.4, 0.35, 0.25],
    #                                                  [0.35, 0.4, 0.35],
    #                                                  [0.25, 0.25, 0.4]],
    #                                          evidence=[commitment_node_0],
    #                                          evidence_card=[3])

    # Associating the CPDs with the network
    model.add_cpds(cpd_commitment, cpd_completes_work, cpd_enthusiastic, cpd_go_above_beyond, cpd_takes_charge,
                   cpd_task_done_on_time, cpd_work_accepted_by_others, cpd_positive_tone, cpd_initiate_conversation,
                   cpd_help_others, cpd_complete_more_tasks, cpd_assigns_tasks, cpd_initiate_meetings,
                   cpd_review_others_work, cpd_mutual_accountability, cpd_trust, commitment_transitional_cpd)

    model.initialize_initial_state()
    return model


#Student A
def student_A(model):
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    week1_result = []
    week2_result = []
    for i in range(0, MAX_ITERATIONS):
        print("week 0")

        sim = model.simulate(1, evidence={commitment_node_0: 1})
        student_1_result_high.iloc[i, 0] = 0
        student_1_result_mid.iloc[i, 0] = 1
        student_1_result_low.iloc[i, 0] = 0

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[commitment_node_1]
        inference_value = dbn_inf.forward_inference([commitment_node_1], sim_dict)
        result = inference_value[commitment_node_1].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week1_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 2)]
        inference_value = dbn_inf.forward_inference([("Commitment", 2)], sim_dict)
        result = inference_value[("Commitment", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0})  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 3)]
        inference_value = dbn_inf.forward_inference([("Commitment", 3)], sim_dict)
        result = inference_value[("Commitment", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        print("week 4")
        sim = model.simulate(n_samples=1, n_time_slices=5, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 4)]
        inference_value = dbn_inf.forward_inference([("Commitment", 4)], sim_dict)
        result = inference_value[("Commitment", 4)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 4] = commitment_high_prob
        student_1_result_mid.iloc[i, 4] = commitment_mid_prob
        student_1_result_low.iloc[i, 4] = commitment_low_prob

        print("week 5")
        sim = model.simulate(n_samples=1, n_time_slices=6, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 5)]
        inference_value = dbn_inf.forward_inference([("Commitment", 5)], sim_dict)
        result = inference_value[("Commitment", 5)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 5] = commitment_high_prob
        student_1_result_mid.iloc[i, 5] = commitment_mid_prob
        student_1_result_low.iloc[i, 5] = commitment_low_prob

        print("week 6")
        sim = model.simulate(n_samples=1, n_time_slices=7, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0})
        # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 6)]
        inference_value = dbn_inf.forward_inference([("Commitment", 6)], sim_dict)
        result = inference_value[("Commitment", 6)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 6] = commitment_high_prob
        student_1_result_mid.iloc[i, 6] = commitment_mid_prob
        student_1_result_low.iloc[i, 6] = commitment_low_prob

        print("week 7")
        sim = model.simulate(n_samples=1, n_time_slices=8, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Help others", 7): 2
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 7)]
        inference_value = dbn_inf.forward_inference([("Commitment", 7)], sim_dict)
        result = inference_value[("Commitment", 7)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 7] = commitment_high_prob
        student_1_result_mid.iloc[i, 7] = commitment_mid_prob
        student_1_result_low.iloc[i, 7] = commitment_low_prob

        print("week 8")
        sim = model.simulate(n_samples=1, n_time_slices=9, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Help others", 7): 2,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 8)]
        inference_value = dbn_inf.forward_inference([("Commitment", 8)], sim_dict)
        result = inference_value[("Commitment", 8)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 8] = commitment_high_prob
        student_1_result_mid.iloc[i, 8] = commitment_mid_prob
        student_1_result_low.iloc[i, 8] = commitment_low_prob

        print("week 9")
        sim = model.simulate(n_samples=1, n_time_slices=10, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Help others", 7): 2,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 9)]
        inference_value = dbn_inf.forward_inference([("Commitment", 9)], sim_dict)
        result = inference_value[("Commitment", 9)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 9] = commitment_high_prob
        student_1_result_mid.iloc[i, 9] = commitment_mid_prob
        student_1_result_low.iloc[i, 9] = commitment_low_prob

        print("week 10")
        sim = model.simulate(n_samples=1, n_time_slices=11, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Help others", 7): 2,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Help others", 10): 2
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 10)]
        inference_value = dbn_inf.forward_inference([("Commitment", 10)], sim_dict)
        result = inference_value[("Commitment", 10)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 10] = commitment_high_prob
        student_1_result_mid.iloc[i, 10] = commitment_mid_prob
        student_1_result_low.iloc[i, 10] = commitment_low_prob

        print("week 11")
        sim = model.simulate(n_samples=1, n_time_slices=12, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Help others", 7): 2,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Help others", 10): 2,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 11)]
        inference_value = dbn_inf.forward_inference([("Commitment", 11)], sim_dict)
        result = inference_value[("Commitment", 11)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 11] = commitment_high_prob
        student_1_result_mid.iloc[i, 11] = commitment_mid_prob
        student_1_result_low.iloc[i, 11] = commitment_low_prob

        print("week 12")
        sim = model.simulate(n_samples=1, n_time_slices=13, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Help others", 7): 2,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Help others", 10): 2,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0,
            ("Task done on time", 12): 0,
            ("Help others", 12): 0,
            ("Completes more tasks", 12): 0
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 12)]
        inference_value = dbn_inf.forward_inference([("Commitment", 12)], sim_dict)
        result = inference_value[("Commitment", 12)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 12] = commitment_high_prob
        student_1_result_mid.iloc[i, 12] = commitment_mid_prob
        student_1_result_low.iloc[i, 12] = commitment_low_prob

        print("week 13")
        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            help_others_node_1: 2,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Help others", 4): 2,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Help others", 7): 2,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Help others", 10): 2,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0,
            ("Task done on time", 12): 0,
            ("Help others", 12): 0,
            ("Completes more tasks", 12): 0,
            ("Task done on time", 13): 0,
            ("Help others", 13): 2,
            ("Completes more tasks", 13): 0
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 13)]
        inference_value = dbn_inf.forward_inference([("Commitment", 13)], sim_dict)
        result = inference_value[("Commitment", 13)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 13] = commitment_high_prob
        student_1_result_mid.iloc[i, 13] = commitment_mid_prob
        student_1_result_low.iloc[i, 13] = commitment_low_prob

        student_1_result_high.to_pickle("studentA_strong_impression_high.p")
        student_1_result_mid.to_pickle("studentA_strong_impression_mid.p")
        student_1_result_low.to_pickle("studentA_strong_impression_strong_impression_low.p")

#Student A1
def student_A1(model):
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))

    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        # r = np.random.rand(1)
        # cpd = model.get_cpds(commitment_node_0).values
        # if r < cpd[0]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 0})
        # elif r < cpd[1]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 1})
        # else:
        #     sim = model.simulate(1, evidence={commitment_node_0: 2})
        student_1_result_high.iloc[i, 0] = 0
        student_1_result_mid.iloc[i, 0] = 1
        student_1_result_low.iloc[i, 0] = 0

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[commitment_node_1]
        inference_value = dbn_inf.forward_inference([commitment_node_1], sim_dict)
        result = inference_value[commitment_node_1].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 2)]
        inference_value = dbn_inf.forward_inference([("Commitment", 2)], sim_dict)
        result = inference_value[("Commitment", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0})  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 3)]
        inference_value = dbn_inf.forward_inference([("Commitment", 3)], sim_dict)
        result = inference_value[("Commitment", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        print("week 4")
        sim = model.simulate(n_samples=1, n_time_slices=5, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 4)]
        inference_value = dbn_inf.forward_inference([("Commitment", 4)], sim_dict)
        result = inference_value[("Commitment", 4)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 4] = commitment_high_prob
        student_1_result_mid.iloc[i, 4] = commitment_mid_prob
        student_1_result_low.iloc[i, 4] = commitment_low_prob

        print("week 5")
        sim = model.simulate(n_samples=1, n_time_slices=6, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 5)]
        inference_value = dbn_inf.forward_inference([("Commitment", 5)], sim_dict)
        result = inference_value[("Commitment", 5)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 5] = commitment_high_prob
        student_1_result_mid.iloc[i, 5] = commitment_mid_prob
        student_1_result_low.iloc[i, 5] = commitment_low_prob

        print("week 6")
        sim = model.simulate(n_samples=1, n_time_slices=7, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0})
        # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 6)]
        inference_value = dbn_inf.forward_inference([("Commitment", 6)], sim_dict)
        result = inference_value[("Commitment", 6)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 6] = commitment_high_prob
        student_1_result_mid.iloc[i, 6] = commitment_mid_prob
        student_1_result_low.iloc[i, 6] = commitment_low_prob

        print("week 7")
        sim = model.simulate(n_samples=1, n_time_slices=8, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 7)]
        inference_value = dbn_inf.forward_inference([("Commitment", 7)], sim_dict)
        result = inference_value[("Commitment", 7)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 7] = commitment_high_prob
        student_1_result_mid.iloc[i, 7] = commitment_mid_prob
        student_1_result_low.iloc[i, 7] = commitment_low_prob

        print("week 8")
        sim = model.simulate(n_samples=1, n_time_slices=9, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 8)]
        inference_value = dbn_inf.forward_inference([("Commitment", 8)], sim_dict)
        result = inference_value[("Commitment", 8)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 8] = commitment_high_prob
        student_1_result_mid.iloc[i, 8] = commitment_mid_prob
        student_1_result_low.iloc[i, 8] = commitment_low_prob

        print("week 9")
        sim = model.simulate(n_samples=1, n_time_slices=10, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 9)]
        inference_value = dbn_inf.forward_inference([("Commitment", 9)], sim_dict)
        result = inference_value[("Commitment", 9)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 9] = commitment_high_prob
        student_1_result_mid.iloc[i, 9] = commitment_mid_prob
        student_1_result_low.iloc[i, 9] = commitment_low_prob

        print("week 10")
        sim = model.simulate(n_samples=1, n_time_slices=11, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 10)]
        inference_value = dbn_inf.forward_inference([("Commitment", 10)], sim_dict)
        result = inference_value[("Commitment", 10)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 10] = commitment_high_prob
        student_1_result_mid.iloc[i, 10] = commitment_mid_prob
        student_1_result_low.iloc[i, 10] = commitment_low_prob

        print("week 11")
        sim = model.simulate(n_samples=1, n_time_slices=12, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 11)]
        inference_value = dbn_inf.forward_inference([("Commitment", 11)], sim_dict)
        result = inference_value[("Commitment", 11)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 11] = commitment_high_prob
        student_1_result_mid.iloc[i, 11] = commitment_mid_prob
        student_1_result_low.iloc[i, 11] = commitment_low_prob

        print("week 12")
        sim = model.simulate(n_samples=1, n_time_slices=13, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0,
            ("Task done on time", 12): 0,
            ("Help others", 12): 0,
            ("Completes more tasks", 12): 0
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 12)]
        inference_value = dbn_inf.forward_inference([("Commitment", 12)], sim_dict)
        result = inference_value[("Commitment", 12)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 12] = commitment_high_prob
        student_1_result_mid.iloc[i, 12] = commitment_mid_prob
        student_1_result_low.iloc[i, 12] = commitment_low_prob

        print("week 13")
        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 0,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0,
            ("Task done on time", 12): 0,
            ("Help others", 12): 0,
            ("Completes more tasks", 12): 0,
            ("Task done on time", 13): 0,
            ("Completes more tasks", 13): 0
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 13)]
        inference_value = dbn_inf.forward_inference([("Commitment", 13)], sim_dict)
        result = inference_value[("Commitment", 13)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 13] = commitment_high_prob
        student_1_result_mid.iloc[i, 13] = commitment_mid_prob
        student_1_result_low.iloc[i, 13] = commitment_low_prob

        student_1_result_high.to_pickle("studentA1_strong_impression_high.p")
        student_1_result_mid.to_pickle("studentA1_strong_impression_mid.p")
        student_1_result_low.to_pickle("studentA1_strong_impression_low.p")

#Student A2
def student_A2(model):
    '''
    tries a dip and see how fast it recovers
    - student did not complete any work due to excusable circumstances in week 9
    :param model:
    :return:
    '''
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))

    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        # r = np.random.rand(1)
        # cpd = model.get_cpds(commitment_node_0).values
        # if r < cpd[0]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 0})
        # elif r < cpd[1]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 1})
        # else:
        #     sim = model.simulate(1, evidence={commitment_node_0: 2})
        student_1_result_high.iloc[i, 0] = 0
        student_1_result_mid.iloc[i, 0] = 1
        student_1_result_low.iloc[i, 0] = 0

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[commitment_node_1]
        inference_value = dbn_inf.forward_inference([commitment_node_1], sim_dict)
        result = inference_value[commitment_node_1].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 2)]
        inference_value = dbn_inf.forward_inference([("Commitment", 2)], sim_dict)
        result = inference_value[("Commitment", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0})  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 3)]
        inference_value = dbn_inf.forward_inference([("Commitment", 3)], sim_dict)
        result = inference_value[("Commitment", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        print("week 4")
        sim = model.simulate(n_samples=1, n_time_slices=5, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 4)]
        inference_value = dbn_inf.forward_inference([("Commitment", 4)], sim_dict)
        result = inference_value[("Commitment", 4)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 4] = commitment_high_prob
        student_1_result_mid.iloc[i, 4] = commitment_mid_prob
        student_1_result_low.iloc[i, 4] = commitment_low_prob

        print("week 5")
        sim = model.simulate(n_samples=1, n_time_slices=6, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 5)]
        inference_value = dbn_inf.forward_inference([("Commitment", 5)], sim_dict)
        result = inference_value[("Commitment", 5)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 5] = commitment_high_prob
        student_1_result_mid.iloc[i, 5] = commitment_mid_prob
        student_1_result_low.iloc[i, 5] = commitment_low_prob

        print("week 6")
        sim = model.simulate(n_samples=1, n_time_slices=7, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0})
        # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 6)]
        inference_value = dbn_inf.forward_inference([("Commitment", 6)], sim_dict)
        result = inference_value[("Commitment", 6)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 6] = commitment_high_prob
        student_1_result_mid.iloc[i, 6] = commitment_mid_prob
        student_1_result_low.iloc[i, 6] = commitment_low_prob

        print("week 7")
        sim = model.simulate(n_samples=1, n_time_slices=8, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 7)]
        inference_value = dbn_inf.forward_inference([("Commitment", 7)], sim_dict)
        result = inference_value[("Commitment", 7)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 7] = commitment_high_prob
        student_1_result_mid.iloc[i, 7] = commitment_mid_prob
        student_1_result_low.iloc[i, 7] = commitment_low_prob

        print("week 8")
        sim = model.simulate(n_samples=1, n_time_slices=9, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 8)]
        inference_value = dbn_inf.forward_inference([("Commitment", 8)], sim_dict)
        result = inference_value[("Commitment", 8)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 8] = commitment_high_prob
        student_1_result_mid.iloc[i, 8] = commitment_mid_prob
        student_1_result_low.iloc[i, 8] = commitment_low_prob

        print("week 9")
        sim = model.simulate(n_samples=1, n_time_slices=10, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 2,
            ("Help others", 9): 0
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 9)]
        inference_value = dbn_inf.forward_inference([("Commitment", 9)], sim_dict)
        result = inference_value[("Commitment", 9)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 9] = commitment_high_prob
        student_1_result_mid.iloc[i, 9] = commitment_mid_prob
        student_1_result_low.iloc[i, 9] = commitment_low_prob

        print("week 10")
        sim = model.simulate(n_samples=1, n_time_slices=11, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 2,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 10)]
        inference_value = dbn_inf.forward_inference([("Commitment", 10)], sim_dict)
        result = inference_value[("Commitment", 10)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 10] = commitment_high_prob
        student_1_result_mid.iloc[i, 10] = commitment_mid_prob
        student_1_result_low.iloc[i, 10] = commitment_low_prob

        print("week 11")
        sim = model.simulate(n_samples=1, n_time_slices=12, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 2,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 11)]
        inference_value = dbn_inf.forward_inference([("Commitment", 11)], sim_dict)
        result = inference_value[("Commitment", 11)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 11] = commitment_high_prob
        student_1_result_mid.iloc[i, 11] = commitment_mid_prob
        student_1_result_low.iloc[i, 11] = commitment_low_prob

        print("week 12")
        sim = model.simulate(n_samples=1, n_time_slices=13, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 2,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0,
            ("Task done on time", 12): 0,
            ("Help others", 12): 0,
            ("Completes more tasks", 12): 0
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 12)]
        inference_value = dbn_inf.forward_inference([("Commitment", 12)], sim_dict)
        result = inference_value[("Commitment", 12)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 12] = commitment_high_prob
        student_1_result_mid.iloc[i, 12] = commitment_mid_prob
        student_1_result_low.iloc[i, 12] = commitment_low_prob

        print("week 13")
        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 0,
            ("Task done on time", 2): 0,
            ("Help others", 2): 1,
            ("Task done on time", 3): 0,
            ("Help others", 3): 0,
            ("Task done on time", 4): 0,
            ("Task done on time", 5): 0,
            ("Help others", 5): 1,
            ("Task done on time", 6): 0,
            ("Help others", 6): 0,
            ("Task done on time", 7): 0,
            ("Task done on time", 8): 0,
            ("Help others", 8): 1,
            ("Task done on time", 9): 2,
            ("Help others", 9): 0,
            ("Task done on time", 10): 0,
            ("Task done on time", 11): 0,
            ("Help others", 11): 1,
            ("Completes more tasks", 11): 0,
            ("Task done on time", 12): 0,
            ("Help others", 12): 0,
            ("Completes more tasks", 12): 0,
            ("Task done on time", 13): 0,
            ("Completes more tasks", 13): 0
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 13)]
        inference_value = dbn_inf.forward_inference([("Commitment", 13)], sim_dict)
        result = inference_value[("Commitment", 13)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 13] = commitment_high_prob
        student_1_result_mid.iloc[i, 13] = commitment_mid_prob
        student_1_result_low.iloc[i, 13] = commitment_low_prob

        student_1_result_high.to_pickle("studentA2_strong_impression_high.p")
        student_1_result_mid.to_pickle("studentA2_strong_impression_mid.p")
        student_1_result_low.to_pickle("studentA2_strong_impression_low.p")

#Student B
def student_B(model):
    '''
    always positive tone, initiate conversation every other week, initiate meeting every 3 weeks
    '''
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))

    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        # r = np.random.rand(1)
        # cpd = model.get_cpds(commitment_node_0).values
        # if r < cpd[0]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 0})
        # elif r < cpd[1]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 1})
        # else:
        #     sim = model.simulate(1, evidence={commitment_node_0: 2})
        student_1_result_high.iloc[i, 0] = 0
        student_1_result_mid.iloc[i, 0] = 1
        student_1_result_low.iloc[i, 0] = 0

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 1)]
        inference_value = dbn_inf.forward_inference([("Commitment", 1)], sim_dict)
        result = inference_value[("Commitment", 1)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 2)]
        inference_value = dbn_inf.forward_inference([("Commitment", 2)], sim_dict)
        result = inference_value[("Commitment", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0})  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 3)]
        inference_value = dbn_inf.forward_inference([("Commitment", 3)], sim_dict)
        result = inference_value[("Commitment", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        print("week 4")
        sim = model.simulate(n_samples=1, n_time_slices=5, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 4)]
        inference_value = dbn_inf.forward_inference([("Commitment", 4)], sim_dict)
        result = inference_value[("Commitment", 4)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 4] = commitment_high_prob
        student_1_result_mid.iloc[i, 4] = commitment_mid_prob
        student_1_result_low.iloc[i, 4] = commitment_low_prob

        print("week 5")
        sim = model.simulate(n_samples=1, n_time_slices=6, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 5)]
        inference_value = dbn_inf.forward_inference([("Commitment", 5)], sim_dict)
        result = inference_value[("Commitment", 5)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 5] = commitment_high_prob
        student_1_result_mid.iloc[i, 5] = commitment_mid_prob
        student_1_result_low.iloc[i, 5] = commitment_low_prob

        print("week 6")
        sim = model.simulate(n_samples=1, n_time_slices=7, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
        })
        # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 6)]
        inference_value = dbn_inf.forward_inference([("Commitment", 6)], sim_dict)
        result = inference_value[("Commitment", 6)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 6] = commitment_high_prob
        student_1_result_mid.iloc[i, 6] = commitment_mid_prob
        student_1_result_low.iloc[i, 6] = commitment_low_prob

        print("week 7")
        sim = model.simulate(n_samples=1, n_time_slices=8, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 7)]
        inference_value = dbn_inf.forward_inference([("Commitment", 7)], sim_dict)
        result = inference_value[("Commitment", 7)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 7] = commitment_high_prob
        student_1_result_mid.iloc[i, 7] = commitment_mid_prob
        student_1_result_low.iloc[i, 7] = commitment_low_prob

        print("week 8")
        sim = model.simulate(n_samples=1, n_time_slices=9, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 8)]
        inference_value = dbn_inf.forward_inference([("Commitment", 8)], sim_dict)
        result = inference_value[("Commitment", 8)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 8] = commitment_high_prob
        student_1_result_mid.iloc[i, 8] = commitment_mid_prob
        student_1_result_low.iloc[i, 8] = commitment_low_prob

        print("week 9")
        sim = model.simulate(n_samples=1, n_time_slices=10, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 9)]
        inference_value = dbn_inf.forward_inference([("Commitment", 9)], sim_dict)
        result = inference_value[("Commitment", 9)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 9] = commitment_high_prob
        student_1_result_mid.iloc[i, 9] = commitment_mid_prob
        student_1_result_low.iloc[i, 9] = commitment_low_prob

        print("week 10")
        sim = model.simulate(n_samples=1, n_time_slices=11, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Positive tone", 10): 0,
            ("Initiate meeting", 10): 0,
        }) # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 10)]
        inference_value = dbn_inf.forward_inference([("Commitment", 10)], sim_dict)
        result = inference_value[("Commitment", 10)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 10] = commitment_high_prob
        student_1_result_mid.iloc[i, 10] = commitment_mid_prob
        student_1_result_low.iloc[i, 10] = commitment_low_prob

        print("week 11")
        sim = model.simulate(n_samples=1, n_time_slices=12, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Positive tone", 10): 0,
            ("Initiate meeting", 10): 0,
            ("Positive tone", 11): 0,
            ("Initiate conversations", 11): 0,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 11)]
        inference_value = dbn_inf.forward_inference([("Commitment", 11)], sim_dict)
        result = inference_value[("Commitment", 11)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 11] = commitment_high_prob
        student_1_result_mid.iloc[i, 11] = commitment_mid_prob
        student_1_result_low.iloc[i, 11] = commitment_low_prob

        print("week 12")
        sim = model.simulate(n_samples=1, n_time_slices=13, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Positive tone", 10): 0,
            ("Initiate meeting", 10): 0,
            ("Positive tone", 11): 0,
            ("Initiate conversations", 11): 0,
            ("Positive tone", 12): 0,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 12)]
        inference_value = dbn_inf.forward_inference([("Commitment", 12)], sim_dict)
        result = inference_value[("Commitment", 12)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 12] = commitment_high_prob
        student_1_result_mid.iloc[i, 12] = commitment_mid_prob
        student_1_result_low.iloc[i, 12] = commitment_low_prob

        print("week 13")
        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Positive tone", 10): 0,
            ("Initiate meeting", 10): 0,
            ("Positive tone", 11): 0,
            ("Initiate conversations", 11): 0,
            ("Positive tone", 12): 0,
            ("Positive tone", 13): 0,
            ("Initiate conversations", 13): 0,
            ("Initiate meeting", 13): 0,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 13)]
        inference_value = dbn_inf.forward_inference([("Commitment", 13)], sim_dict)
        result = inference_value[("Commitment", 13)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 13] = commitment_high_prob
        student_1_result_mid.iloc[i, 13] = commitment_mid_prob
        student_1_result_low.iloc[i, 13] = commitment_low_prob

        student_1_result_high.to_pickle("studentB_strong_impression_high.p")
        student_1_result_mid.to_pickle("studentB_strong_impression_mid.p")
        student_1_result_low.to_pickle("studentB_strong_impression_low.p")


def student_B1(model):
    '''
    always positive tone, initiate conversation every other week, initiate meeting every 3 weeks
    BUT never completes any tasks on time
    '''
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))

    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        # r = np.random.rand(1)
        # cpd = model.get_cpds(commitment_node_0).values
        # if r < cpd[0]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 0})
        # elif r < cpd[1]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 1})
        # else:
        #     sim = model.simulate(1, evidence={commitment_node_0: 2})
        student_1_result_high.iloc[i, 0] = 0
        student_1_result_mid.iloc[i, 0] = 1
        student_1_result_low.iloc[i, 0] = 0

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 1)]
        inference_value = dbn_inf.forward_inference([("Commitment", 1)], sim_dict)
        result = inference_value[("Commitment", 1)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Task done on time", 2): 2})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 2)]
        inference_value = dbn_inf.forward_inference([("Commitment", 2)], sim_dict)
        result = inference_value[("Commitment", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2})  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 3)]
        inference_value = dbn_inf.forward_inference([("Commitment", 3)], sim_dict)
        result = inference_value[("Commitment", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        print("week 4")
        sim = model.simulate(n_samples=1, n_time_slices=5, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 4)]
        inference_value = dbn_inf.forward_inference([("Commitment", 4)], sim_dict)
        result = inference_value[("Commitment", 4)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 4] = commitment_high_prob
        student_1_result_mid.iloc[i, 4] = commitment_mid_prob
        student_1_result_low.iloc[i, 4] = commitment_low_prob

        print("week 5")
        sim = model.simulate(n_samples=1, n_time_slices=6, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 5)]
        inference_value = dbn_inf.forward_inference([("Commitment", 5)], sim_dict)
        result = inference_value[("Commitment", 5)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 5] = commitment_high_prob
        student_1_result_mid.iloc[i, 5] = commitment_mid_prob
        student_1_result_low.iloc[i, 5] = commitment_low_prob

        print("week 6")
        sim = model.simulate(n_samples=1, n_time_slices=7, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
            ("Task done on time", 6): 2,
        })
        # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 6)]
        inference_value = dbn_inf.forward_inference([("Commitment", 6)], sim_dict)
        result = inference_value[("Commitment", 6)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 6] = commitment_high_prob
        student_1_result_mid.iloc[i, 6] = commitment_mid_prob
        student_1_result_low.iloc[i, 6] = commitment_low_prob

        print("week 7")
        sim = model.simulate(n_samples=1, n_time_slices=8, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
            ("Task done on time", 6): 2,
            ("Task done on time", 7): 2,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 7)]
        inference_value = dbn_inf.forward_inference([("Commitment", 7)], sim_dict)
        result = inference_value[("Commitment", 7)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 7] = commitment_high_prob
        student_1_result_mid.iloc[i, 7] = commitment_mid_prob
        student_1_result_low.iloc[i, 7] = commitment_low_prob

        print("week 8")
        sim = model.simulate(n_samples=1, n_time_slices=9, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
            ("Task done on time", 6): 2,
            ("Task done on time", 7): 2,
            ("Task done on time", 8): 2,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 8)]
        inference_value = dbn_inf.forward_inference([("Commitment", 8)], sim_dict)
        result = inference_value[("Commitment", 8)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 8] = commitment_high_prob
        student_1_result_mid.iloc[i, 8] = commitment_mid_prob
        student_1_result_low.iloc[i, 8] = commitment_low_prob

        print("week 9")
        sim = model.simulate(n_samples=1, n_time_slices=10, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
            ("Task done on time", 6): 2,
            ("Task done on time", 7): 2,
            ("Task done on time", 8): 2,
            ("Task done on time", 9): 2,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 9)]
        inference_value = dbn_inf.forward_inference([("Commitment", 9)], sim_dict)
        result = inference_value[("Commitment", 9)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 9] = commitment_high_prob
        student_1_result_mid.iloc[i, 9] = commitment_mid_prob
        student_1_result_low.iloc[i, 9] = commitment_low_prob

        print("week 10")
        sim = model.simulate(n_samples=1, n_time_slices=11, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Positive tone", 10): 0,
            ("Initiate meeting", 10): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
            ("Task done on time", 6): 2,
            ("Task done on time", 7): 2,
            ("Task done on time", 8): 2,
            ("Task done on time", 9): 2,
            ("Task done on time", 10): 2,
        }) # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 10)]
        inference_value = dbn_inf.forward_inference([("Commitment", 10)], sim_dict)
        result = inference_value[("Commitment", 10)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 10] = commitment_high_prob
        student_1_result_mid.iloc[i, 10] = commitment_mid_prob
        student_1_result_low.iloc[i, 10] = commitment_low_prob

        print("week 11")
        sim = model.simulate(n_samples=1, n_time_slices=12, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Positive tone", 10): 0,
            ("Initiate meeting", 10): 0,
            ("Positive tone", 11): 0,
            ("Initiate conversations", 11): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
            ("Task done on time", 6): 2,
            ("Task done on time", 7): 2,
            ("Task done on time", 8): 2,
            ("Task done on time", 9): 2,
            ("Task done on time", 10): 2,
            ("Task done on time", 11): 2,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 11)]
        inference_value = dbn_inf.forward_inference([("Commitment", 11)], sim_dict)
        result = inference_value[("Commitment", 11)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 11] = commitment_high_prob
        student_1_result_mid.iloc[i, 11] = commitment_mid_prob
        student_1_result_low.iloc[i, 11] = commitment_low_prob

        print("week 12")
        sim = model.simulate(n_samples=1, n_time_slices=13, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Positive tone", 10): 0,
            ("Initiate meeting", 10): 0,
            ("Positive tone", 11): 0,
            ("Initiate conversations", 11): 0,
            ("Positive tone", 12): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
            ("Task done on time", 6): 2,
            ("Task done on time", 7): 2,
            ("Task done on time", 8): 2,
            ("Task done on time", 9): 2,
            ("Task done on time", 10): 2,
            ("Task done on time", 11): 2,
            ("Task done on time", 12): 2,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 12)]
        inference_value = dbn_inf.forward_inference([("Commitment", 12)], sim_dict)
        result = inference_value[("Commitment", 12)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 12] = commitment_high_prob
        student_1_result_mid.iloc[i, 12] = commitment_mid_prob
        student_1_result_low.iloc[i, 12] = commitment_low_prob

        print("week 13")
        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={
            # commitment_node_0: 1,
            positive_tone_node_1: 0,
            initiate_conversations_node_1: 0,
            initiate_meeting_node_1: 0,
            task_done_on_time_node_1: 2,
            ("Positive tone", 2): 0,
            ("Positive tone", 3): 0,
            ("Initiate conversations", 3): 0,
            ("Positive tone", 4): 0,
            ("Initiate meeting", 4): 0,
            ("Positive tone", 5): 0,
            ("Initiate conversations", 5): 0,
            ("Positive tone", 6): 0,
            ("Positive tone", 7): 0,
            ("Initiate conversations", 7): 0,
            ("Initiate meeting", 7): 0,
            ("Positive tone", 8): 0,
            ("Positive tone", 9): 0,
            ("Initiate conversations", 9): 0,
            ("Positive tone", 10): 0,
            ("Initiate meeting", 10): 0,
            ("Positive tone", 11): 0,
            ("Initiate conversations", 11): 0,
            ("Positive tone", 12): 0,
            ("Positive tone", 13): 0,
            ("Initiate conversations", 13): 0,
            ("Initiate meeting", 13): 0,
            ("Task done on time", 2): 2,
            ("Task done on time", 3): 2,
            ("Task done on time", 4): 2,
            ("Task done on time", 5): 2,
            ("Task done on time", 6): 2,
            ("Task done on time", 7): 2,
            ("Task done on time", 8): 2,
            ("Task done on time", 9): 2,
            ("Task done on time", 10): 2,
            ("Task done on time", 11): 2,
            ("Task done on time", 12): 2,
            ("Task done on time", 13): 2,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 13)]
        inference_value = dbn_inf.forward_inference([("Commitment", 13)], sim_dict)
        result = inference_value[("Commitment", 13)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 13] = commitment_high_prob
        student_1_result_mid.iloc[i, 13] = commitment_mid_prob
        student_1_result_low.iloc[i, 13] = commitment_low_prob

        student_1_result_high.to_pickle("studentB1_strong_impression_high.p")
        student_1_result_mid.to_pickle("studentB1_strong_impression_mid.p")
        student_1_result_low.to_pickle("studentB1_strong_impression_low.p")

        #TODO: change file name of student B to B1 then re-run student B

# Student C
def student_C(model):
    '''
    Only gets partial task done on time every week, shallow helps others every 3 weeks
    '''
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))

    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        # r = np.random.rand(1)
        # cpd = model.get_cpds(commitment_node_0).values
        # if r < cpd[0]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 0})
        # elif r < cpd[1]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 1})
        # else:
        #     sim = model.simulate(1, evidence={commitment_node_0: 2})
        student_1_result_high.iloc[i, 0] = 0
        student_1_result_mid.iloc[i, 0] = 1
        student_1_result_low.iloc[i, 0] = 0

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1})  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 1)]
        inference_value = dbn_inf.forward_inference([("Commitment", 1)], sim_dict)
        result = inference_value[("Commitment", 1)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1})  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 2)]
        inference_value = dbn_inf.forward_inference([("Commitment", 2)], sim_dict)
        result = inference_value[("Commitment", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1})  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 3)]
        inference_value = dbn_inf.forward_inference([("Commitment", 3)], sim_dict)
        result = inference_value[("Commitment", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        print("week 4")
        sim = model.simulate(n_samples=1, n_time_slices=5, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 4)]
        inference_value = dbn_inf.forward_inference([("Commitment", 4)], sim_dict)
        result = inference_value[("Commitment", 4)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 4] = commitment_high_prob
        student_1_result_mid.iloc[i, 4] = commitment_mid_prob
        student_1_result_low.iloc[i, 4] = commitment_low_prob

        print("week 5")
        sim = model.simulate(n_samples=1, n_time_slices=6, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 5)]
        inference_value = dbn_inf.forward_inference([("Commitment", 5)], sim_dict)
        result = inference_value[("Commitment", 5)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 5] = commitment_high_prob
        student_1_result_mid.iloc[i, 5] = commitment_mid_prob
        student_1_result_low.iloc[i, 5] = commitment_low_prob

        print("week 6")
        sim = model.simulate(n_samples=1, n_time_slices=7, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
            ("Task done on time", 6): 1,
        })
        # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 6)]
        inference_value = dbn_inf.forward_inference([("Commitment", 6)], sim_dict)
        result = inference_value[("Commitment", 6)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 6] = commitment_high_prob
        student_1_result_mid.iloc[i, 6] = commitment_mid_prob
        student_1_result_low.iloc[i, 6] = commitment_low_prob

        print("week 7")
        sim = model.simulate(n_samples=1, n_time_slices=8, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
            ("Task done on time", 6): 1,
            ("Task done on time", 7): 1,
            ("Help others", 7): 1
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 7)]
        inference_value = dbn_inf.forward_inference([("Commitment", 7)], sim_dict)
        result = inference_value[("Commitment", 7)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 7] = commitment_high_prob
        student_1_result_mid.iloc[i, 7] = commitment_mid_prob
        student_1_result_low.iloc[i, 7] = commitment_low_prob

        print("week 8")
        sim = model.simulate(n_samples=1, n_time_slices=9, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
            ("Task done on time", 6): 1,
            ("Task done on time", 7): 1,
            ("Help others", 7): 1,
            ("Task done on time", 8): 1,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 8)]
        inference_value = dbn_inf.forward_inference([("Commitment", 8)], sim_dict)
        result = inference_value[("Commitment", 8)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 8] = commitment_high_prob
        student_1_result_mid.iloc[i, 8] = commitment_mid_prob
        student_1_result_low.iloc[i, 8] = commitment_low_prob

        print("week 9")
        sim = model.simulate(n_samples=1, n_time_slices=10, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
            ("Task done on time", 6): 1,
            ("Task done on time", 7): 1,
            ("Help others", 7): 1,
            ("Task done on time", 8): 1,
            ("Task done on time", 9): 1,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 9)]
        inference_value = dbn_inf.forward_inference([("Commitment", 9)], sim_dict)
        result = inference_value[("Commitment", 9)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 9] = commitment_high_prob
        student_1_result_mid.iloc[i, 9] = commitment_mid_prob
        student_1_result_low.iloc[i, 9] = commitment_low_prob

        print("week 10")
        sim = model.simulate(n_samples=1, n_time_slices=11, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
            ("Task done on time", 6): 1,
            ("Task done on time", 7): 1,
            ("Help others", 7): 1,
            ("Task done on time", 8): 1,
            ("Task done on time", 9): 1,
            ("Task done on time", 10): 1,
            ("Help others", 10): 1,
        }) # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 10)]
        inference_value = dbn_inf.forward_inference([("Commitment", 10)], sim_dict)
        result = inference_value[("Commitment", 10)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 10] = commitment_high_prob
        student_1_result_mid.iloc[i, 10] = commitment_mid_prob
        student_1_result_low.iloc[i, 10] = commitment_low_prob

        print("week 11")
        sim = model.simulate(n_samples=1, n_time_slices=12, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
            ("Task done on time", 6): 1,
            ("Task done on time", 7): 1,
            ("Help others", 7): 1,
            ("Task done on time", 8): 1,
            ("Task done on time", 9): 1,
            ("Task done on time", 10): 1,
            ("Help others", 10): 1,
            ("Task done on time", 11): 1,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 11)]
        inference_value = dbn_inf.forward_inference([("Commitment", 11)], sim_dict)
        result = inference_value[("Commitment", 11)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 11] = commitment_high_prob
        student_1_result_mid.iloc[i, 11] = commitment_mid_prob
        student_1_result_low.iloc[i, 11] = commitment_low_prob

        print("week 12")
        sim = model.simulate(n_samples=1, n_time_slices=13, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
            ("Task done on time", 6): 1,
            ("Task done on time", 7): 1,
            ("Help others", 7): 1,
            ("Task done on time", 8): 1,
            ("Task done on time", 9): 1,
            ("Task done on time", 10): 1,
            ("Help others", 10): 1,
            ("Task done on time", 11): 1,
            ("Task done on time", 12): 1,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 12)]
        inference_value = dbn_inf.forward_inference([("Commitment", 12)], sim_dict)
        result = inference_value[("Commitment", 12)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 12] = commitment_high_prob
        student_1_result_mid.iloc[i, 12] = commitment_mid_prob
        student_1_result_low.iloc[i, 12] = commitment_low_prob

        print("week 13")
        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            help_others_node_1: 1,
            ("Task done on time", 2): 1,
            ("Task done on time", 3): 1,
            ("Task done on time", 4): 1,
            ("Help others", 4): 1,
            ("Task done on time", 5): 1,
            ("Task done on time", 6): 1,
            ("Task done on time", 7): 1,
            ("Help others", 7): 1,
            ("Task done on time", 8): 1,
            ("Task done on time", 9): 1,
            ("Task done on time", 10): 1,
            ("Help others", 10): 1,
            ("Task done on time", 11): 1,
            ("Task done on time", 12): 1,
            ("Task done on time", 13): 1,
            ("Help others", 13): 1,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 13)]
        inference_value = dbn_inf.forward_inference([("Commitment", 13)], sim_dict)
        result = inference_value[("Commitment", 13)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 13] = commitment_high_prob
        student_1_result_mid.iloc[i, 13] = commitment_mid_prob
        student_1_result_low.iloc[i, 13] = commitment_low_prob

        student_1_result_high.to_pickle("studentC_strong_impression_high.p")
        student_1_result_mid.to_pickle("studentC_strong_impression_mid.p")
        student_1_result_low.to_pickle("studentC_strong_impression_low.p")


def student_D(model):
    '''
    Only gets partial task done every odd week, gets no task done every even week,
    work only ever partially accepted by others, and never initiate conversations
    '''
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))

    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        # r = np.random.rand(1)
        # cpd = model.get_cpds(commitment_node_0).values
        # if r < cpd[0]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 0})
        # elif r < cpd[1]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 1})
        # else:
        #     sim = model.simulate(1, evidence={commitment_node_0: 2})
        student_1_result_high.iloc[i, 0] = 0
        student_1_result_mid.iloc[i, 0] = 1
        student_1_result_low.iloc[i, 0] = 0

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 1)]
        inference_value = dbn_inf.forward_inference([("Commitment", 1)], sim_dict)
        result = inference_value[("Commitment", 1)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 2)]
        inference_value = dbn_inf.forward_inference([("Commitment", 2)], sim_dict)
        result = inference_value[("Commitment", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 3)]
        inference_value = dbn_inf.forward_inference([("Commitment", 3)], sim_dict)
        result = inference_value[("Commitment", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        print("week 4")
        sim = model.simulate(n_samples=1, n_time_slices=5, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 4)]
        inference_value = dbn_inf.forward_inference([("Commitment", 4)], sim_dict)
        result = inference_value[("Commitment", 4)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 4] = commitment_high_prob
        student_1_result_mid.iloc[i, 4] = commitment_mid_prob
        student_1_result_low.iloc[i, 4] = commitment_low_prob

        print("week 5")
        sim = model.simulate(n_samples=1, n_time_slices=6, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 5)]
        inference_value = dbn_inf.forward_inference([("Commitment", 5)], sim_dict)
        result = inference_value[("Commitment", 5)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 5] = commitment_high_prob
        student_1_result_mid.iloc[i, 5] = commitment_mid_prob
        student_1_result_low.iloc[i, 5] = commitment_low_prob

        print("week 6")
        sim = model.simulate(n_samples=1, n_time_slices=7, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
            ("Task done on time", 6): 2,
            ("Work accepted by others", 6): 1,
            ("Initiate conversations", 6): 1,
        })
        # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 6)]
        inference_value = dbn_inf.forward_inference([("Commitment", 6)], sim_dict)
        result = inference_value[("Commitment", 6)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 6] = commitment_high_prob
        student_1_result_mid.iloc[i, 6] = commitment_mid_prob
        student_1_result_low.iloc[i, 6] = commitment_low_prob

        print("week 7")
        sim = model.simulate(n_samples=1, n_time_slices=8, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
            ("Task done on time", 6): 2,
            ("Work accepted by others", 6): 1,
            ("Initiate conversations", 6): 1,
            ("Task done on time", 7): 1,
            ("Work accepted by others", 7): 1,
            ("Initiate conversations", 7): 1,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 7)]
        inference_value = dbn_inf.forward_inference([("Commitment", 7)], sim_dict)
        result = inference_value[("Commitment", 7)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 7] = commitment_high_prob
        student_1_result_mid.iloc[i, 7] = commitment_mid_prob
        student_1_result_low.iloc[i, 7] = commitment_low_prob

        print("week 8")
        sim = model.simulate(n_samples=1, n_time_slices=9, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
            ("Task done on time", 6): 2,
            ("Work accepted by others", 6): 1,
            ("Initiate conversations", 6): 1,
            ("Task done on time", 7): 1,
            ("Work accepted by others", 7): 1,
            ("Initiate conversations", 7): 1,
            ("Task done on time", 8): 2,
            ("Work accepted by others", 8): 1,
            ("Initiate conversations", 8): 1,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 8)]
        inference_value = dbn_inf.forward_inference([("Commitment", 8)], sim_dict)
        result = inference_value[("Commitment", 8)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 8] = commitment_high_prob
        student_1_result_mid.iloc[i, 8] = commitment_mid_prob
        student_1_result_low.iloc[i, 8] = commitment_low_prob

        print("week 9")
        sim = model.simulate(n_samples=1, n_time_slices=10, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
            ("Task done on time", 6): 2,
            ("Work accepted by others", 6): 1,
            ("Initiate conversations", 6): 1,
            ("Task done on time", 7): 1,
            ("Work accepted by others", 7): 1,
            ("Initiate conversations", 7): 1,
            ("Task done on time", 8): 2,
            ("Work accepted by others", 8): 1,
            ("Initiate conversations", 8): 1,
            ("Task done on time", 9): 1,
            ("Work accepted by others", 9): 1,
            ("Initiate conversations", 9): 1,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 9)]
        inference_value = dbn_inf.forward_inference([("Commitment", 9)], sim_dict)
        result = inference_value[("Commitment", 9)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 9] = commitment_high_prob
        student_1_result_mid.iloc[i, 9] = commitment_mid_prob
        student_1_result_low.iloc[i, 9] = commitment_low_prob

        print("week 10")
        sim = model.simulate(n_samples=1, n_time_slices=11, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
            ("Task done on time", 6): 2,
            ("Work accepted by others", 6): 1,
            ("Initiate conversations", 6): 1,
            ("Task done on time", 7): 1,
            ("Work accepted by others", 7): 1,
            ("Initiate conversations", 7): 1,
            ("Task done on time", 8): 2,
            ("Work accepted by others", 8): 1,
            ("Initiate conversations", 8): 1,
            ("Task done on time", 9): 1,
            ("Work accepted by others", 9): 1,
            ("Initiate conversations", 9): 1,
            ("Task done on time", 10): 2,
            ("Work accepted by others", 10): 1,
            ("Initiate conversations", 10): 1,
        }) # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 10)]
        inference_value = dbn_inf.forward_inference([("Commitment", 10)], sim_dict)
        result = inference_value[("Commitment", 10)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 10] = commitment_high_prob
        student_1_result_mid.iloc[i, 10] = commitment_mid_prob
        student_1_result_low.iloc[i, 10] = commitment_low_prob

        print("week 11")
        sim = model.simulate(n_samples=1, n_time_slices=12, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
            ("Task done on time", 6): 2,
            ("Work accepted by others", 6): 1,
            ("Initiate conversations", 6): 1,
            ("Task done on time", 7): 1,
            ("Work accepted by others", 7): 1,
            ("Initiate conversations", 7): 1,
            ("Task done on time", 8): 2,
            ("Work accepted by others", 8): 1,
            ("Initiate conversations", 8): 1,
            ("Task done on time", 9): 1,
            ("Work accepted by others", 9): 1,
            ("Initiate conversations", 9): 1,
            ("Task done on time", 10): 2,
            ("Work accepted by others", 10): 1,
            ("Initiate conversations", 10): 1,
            ("Task done on time", 11): 1,
            ("Work accepted by others", 11): 1,
            ("Initiate conversations", 11): 1,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 11)]
        inference_value = dbn_inf.forward_inference([("Commitment", 11)], sim_dict)
        result = inference_value[("Commitment", 11)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 11] = commitment_high_prob
        student_1_result_mid.iloc[i, 11] = commitment_mid_prob
        student_1_result_low.iloc[i, 11] = commitment_low_prob

        print("week 12")
        sim = model.simulate(n_samples=1, n_time_slices=13, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
            ("Task done on time", 6): 2,
            ("Work accepted by others", 6): 1,
            ("Initiate conversations", 6): 1,
            ("Task done on time", 7): 1,
            ("Work accepted by others", 7): 1,
            ("Initiate conversations", 7): 1,
            ("Task done on time", 8): 2,
            ("Work accepted by others", 8): 1,
            ("Initiate conversations", 8): 1,
            ("Task done on time", 9): 1,
            ("Work accepted by others", 9): 1,
            ("Initiate conversations", 9): 1,
            ("Task done on time", 10): 2,
            ("Work accepted by others", 10): 1,
            ("Initiate conversations", 10): 1,
            ("Task done on time", 11): 1,
            ("Work accepted by others", 11): 1,
            ("Initiate conversations", 11): 1,
            ("Task done on time", 12): 2,
            ("Work accepted by others", 12): 1,
            ("Initiate conversations", 12): 1,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 12)]
        inference_value = dbn_inf.forward_inference([("Commitment", 12)], sim_dict)
        result = inference_value[("Commitment", 12)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 12] = commitment_high_prob
        student_1_result_mid.iloc[i, 12] = commitment_mid_prob
        student_1_result_low.iloc[i, 12] = commitment_low_prob

        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 1,
            work_accepted_by_others_node_1: 1,
            initiate_conversations_node_1: 1,
            ("Task done on time", 2): 2,
            ("Work accepted by others", 2): 1,
            ("Initiate conversations", 2): 1,
            ("Task done on time", 3): 1,
            ("Work accepted by others", 3): 1,
            ("Initiate conversations", 3): 1,
            ("Task done on time", 4): 2,
            ("Work accepted by others", 4): 1,
            ("Initiate conversations", 4): 1,
            ("Task done on time", 5): 1,
            ("Work accepted by others", 5): 1,
            ("Initiate conversations", 5): 1,
            ("Task done on time", 6): 2,
            ("Work accepted by others", 6): 1,
            ("Initiate conversations", 6): 1,
            ("Task done on time", 7): 1,
            ("Work accepted by others", 7): 1,
            ("Initiate conversations", 7): 1,
            ("Task done on time", 8): 2,
            ("Work accepted by others", 8): 1,
            ("Initiate conversations", 8): 1,
            ("Task done on time", 9): 1,
            ("Work accepted by others", 9): 1,
            ("Initiate conversations", 9): 1,
            ("Task done on time", 10): 2,
            ("Work accepted by others", 10): 1,
            ("Initiate conversations", 10): 1,
            ("Task done on time", 11): 1,
            ("Work accepted by others", 11): 1,
            ("Initiate conversations", 11): 1,
            ("Task done on time", 12): 2,
            ("Work accepted by others", 12): 1,
            ("Initiate conversations", 12): 1,
            ("Task done on time", 13): 2,
            ("Work accepted by others", 13): 1,
            ("Initiate conversations", 13): 1,
        })  # task done on time _all help others no observation
        print("week 13")
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 13)]
        inference_value = dbn_inf.forward_inference([("Commitment", 13)], sim_dict)
        result = inference_value[("Commitment", 13)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 13] = commitment_high_prob
        student_1_result_mid.iloc[i, 13] = commitment_mid_prob
        student_1_result_low.iloc[i, 13] = commitment_low_prob

        student_1_result_high.to_pickle("studentD_strong_impression_high.p")
        student_1_result_mid.to_pickle("studentD_strong_impression_mid.p")
        student_1_result_low.to_pickle("studentD_strong_impression_low.p")


def student_F(model):
    '''
    Never gets work done, always negative attitude
    '''
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 14), index=range(0, MAX_ITERATIONS))

    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        # r = np.random.rand(1)
        # cpd = model.get_cpds(commitment_node_0).values
        # if r < cpd[0]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 0})
        # elif r < cpd[1]:
        #     sim = model.simulate(1, evidence={commitment_node_0: 1})
        # else:
        #     sim = model.simulate(1, evidence={commitment_node_0: 2})
        student_1_result_high.iloc[i, 0] = 0
        student_1_result_mid.iloc[i, 0] = 1
        student_1_result_low.iloc[i, 0] = 0

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 1)]
        inference_value = dbn_inf.forward_inference([("Commitment", 1)], sim_dict)
        result = inference_value[("Commitment", 1)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 2)]
        inference_value = dbn_inf.forward_inference([("Commitment", 2)], sim_dict)
        result = inference_value[("Commitment", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 3)]
        inference_value = dbn_inf.forward_inference([("Commitment", 3)], sim_dict)
        result = inference_value[("Commitment", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        print("week 4")
        sim = model.simulate(n_samples=1, n_time_slices=5, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 4)]
        inference_value = dbn_inf.forward_inference([("Commitment", 4)], sim_dict)
        result = inference_value[("Commitment", 4)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 4] = commitment_high_prob
        student_1_result_mid.iloc[i, 4] = commitment_mid_prob
        student_1_result_low.iloc[i, 4] = commitment_low_prob

        print("week 5")
        sim = model.simulate(n_samples=1, n_time_slices=6, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 5)]
        inference_value = dbn_inf.forward_inference([("Commitment", 5)], sim_dict)
        result = inference_value[("Commitment", 5)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 5] = commitment_high_prob
        student_1_result_mid.iloc[i, 5] = commitment_mid_prob
        student_1_result_low.iloc[i, 5] = commitment_low_prob

        print("week 6")
        sim = model.simulate(n_samples=1, n_time_slices=7, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
            ("Task done on time", 6): 2,
            ("Positive tone", 6): 2,
        })
        # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 6)]
        inference_value = dbn_inf.forward_inference([("Commitment", 6)], sim_dict)
        result = inference_value[("Commitment", 6)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 6] = commitment_high_prob
        student_1_result_mid.iloc[i, 6] = commitment_mid_prob
        student_1_result_low.iloc[i, 6] = commitment_low_prob

        print("week 7")
        sim = model.simulate(n_samples=1, n_time_slices=8, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
            ("Task done on time", 6): 2,
            ("Positive tone", 6): 2,
            ("Task done on time", 7): 2,
            ("Positive tone", 7): 2,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 7)]
        inference_value = dbn_inf.forward_inference([("Commitment", 7)], sim_dict)
        result = inference_value[("Commitment", 7)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 7] = commitment_high_prob
        student_1_result_mid.iloc[i, 7] = commitment_mid_prob
        student_1_result_low.iloc[i, 7] = commitment_low_prob

        print("week 8")
        sim = model.simulate(n_samples=1, n_time_slices=9, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
            ("Task done on time", 6): 2,
            ("Positive tone", 6): 2,
            ("Task done on time", 7): 2,
            ("Positive tone", 7): 2,
            ("Task done on time", 8): 2,
            ("Positive tone", 8): 2,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 8)]
        inference_value = dbn_inf.forward_inference([("Commitment", 8)], sim_dict)
        result = inference_value[("Commitment", 8)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 8] = commitment_high_prob
        student_1_result_mid.iloc[i, 8] = commitment_mid_prob
        student_1_result_low.iloc[i, 8] = commitment_low_prob

        print("week 9")
        sim = model.simulate(n_samples=1, n_time_slices=10, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
            ("Task done on time", 6): 2,
            ("Positive tone", 6): 2,
            ("Task done on time", 7): 2,
            ("Positive tone", 7): 2,
            ("Task done on time", 8): 2,
            ("Positive tone", 8): 2,
            ("Task done on time", 9): 2,
            ("Positive tone", 9): 2,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 9)]
        inference_value = dbn_inf.forward_inference([("Commitment", 9)], sim_dict)
        result = inference_value[("Commitment", 9)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 9] = commitment_high_prob
        student_1_result_mid.iloc[i, 9] = commitment_mid_prob
        student_1_result_low.iloc[i, 9] = commitment_low_prob

        print("week 10")
        sim = model.simulate(n_samples=1, n_time_slices=11, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
            ("Task done on time", 6): 2,
            ("Positive tone", 6): 2,
            ("Task done on time", 7): 2,
            ("Positive tone", 7): 2,
            ("Task done on time", 8): 2,
            ("Positive tone", 8): 2,
            ("Task done on time", 9): 2,
            ("Positive tone", 9): 2,
            ("Task done on time", 10): 2,
            ("Positive tone", 10): 2,
        }) # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 10)]
        inference_value = dbn_inf.forward_inference([("Commitment", 10)], sim_dict)
        result = inference_value[("Commitment", 10)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 10] = commitment_high_prob
        student_1_result_mid.iloc[i, 10] = commitment_mid_prob
        student_1_result_low.iloc[i, 10] = commitment_low_prob

        print("week 11")
        sim = model.simulate(n_samples=1, n_time_slices=12, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
            ("Task done on time", 6): 2,
            ("Positive tone", 6): 2,
            ("Task done on time", 7): 2,
            ("Positive tone", 7): 2,
            ("Task done on time", 8): 2,
            ("Positive tone", 8): 2,
            ("Task done on time", 9): 2,
            ("Positive tone", 9): 2,
            ("Task done on time", 10): 2,
            ("Positive tone", 10): 2,
            ("Task done on time", 11): 2,
            ("Positive tone", 11): 2,
        })  # task done on time _all help others shallow
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 11)]
        inference_value = dbn_inf.forward_inference([("Commitment", 11)], sim_dict)
        result = inference_value[("Commitment", 11)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 11] = commitment_high_prob
        student_1_result_mid.iloc[i, 11] = commitment_mid_prob
        student_1_result_low.iloc[i, 11] = commitment_low_prob

        print("week 12")
        sim = model.simulate(n_samples=1, n_time_slices=13, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
            ("Task done on time", 6): 2,
            ("Positive tone", 6): 2,
            ("Task done on time", 7): 2,
            ("Positive tone", 7): 2,
            ("Task done on time", 8): 2,
            ("Positive tone", 8): 2,
            ("Task done on time", 9): 2,
            ("Positive tone", 9): 2,
            ("Task done on time", 10): 2,
            ("Positive tone", 10): 2,
            ("Task done on time", 11): 2,
            ("Positive tone", 11): 2,
            ("Task done on time", 12): 2,
            ("Positive tone", 12): 2,
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 12)]
        inference_value = dbn_inf.forward_inference([("Commitment", 12)], sim_dict)
        result = inference_value[("Commitment", 12)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 12] = commitment_high_prob
        student_1_result_mid.iloc[i, 12] = commitment_mid_prob
        student_1_result_low.iloc[i, 12] = commitment_low_prob

        print("week 13")
        sim = model.simulate(n_samples=1, n_time_slices=14, evidence={
            # commitment_node_0: 1,
            task_done_on_time_node_1: 2,
            positive_tone_node_1: 2,
            ("Task done on time", 2): 2,
            ("Positive tone", 2): 2,
            ("Task done on time", 3): 2,
            ("Positive tone", 3): 2,
            ("Task done on time", 4): 2,
            ("Positive tone", 4): 2,
            ("Task done on time", 5): 2,
            ("Positive tone", 5): 2,
            ("Task done on time", 6): 2,
            ("Positive tone", 6): 2,
            ("Task done on time", 7): 2,
            ("Positive tone", 7): 2,
            ("Task done on time", 8): 2,
            ("Positive tone", 8): 2,
            ("Task done on time", 9): 2,
            ("Positive tone", 9): 2,
            ("Task done on time", 10): 2,
            ("Positive tone", 10): 2,
            ("Task done on time", 11): 2,
            ("Positive tone", 11): 2,
            ("Task done on time", 12): 2,
            ("Positive tone", 12): 2,
            ("Task done on time", 13): 2,
            ("Positive tone", 13): 2,
        })  # task done on time _all help others no observation
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Commitment", 13)]
        inference_value = dbn_inf.forward_inference([("Commitment", 13)], sim_dict)
        result = inference_value[("Commitment", 13)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        student_1_result_high.iloc[i, 13] = commitment_high_prob
        student_1_result_mid.iloc[i, 13] = commitment_mid_prob
        student_1_result_low.iloc[i, 13] = commitment_low_prob

        student_1_result_high.to_pickle("studentF_strong_impression_high.p")
        student_1_result_mid.to_pickle("studentF_strong_impression_mid.p")
        student_1_result_low.to_pickle("studentF_strong_impression_low.p")


if __name__ == '__main__':
    commitment_model = commitment()
    # student_A(commitment_model)
    # student_A1(commitment_model)
    # student_A2(commitment_model)
    # student_B(commitment_model)
    # student_C(commitment_model)
    # student_B1(commitment_model)
    # student_D(commitment_model)
    # student_F(commitment_model)



