import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

agreed_upon_system_node_0 = ("Agreed upon system", 0)
clarity_of_goal_node_0 = ("Clarity of goal", 0)
clarity_of_role_node_0 = ("Clarity of role", 0)
specific_timeline_node_0 = ("Specific timeline", 0)
clarity_of_task_node_0 = ("Clarity of task", 0)
follow_structured_review_system_node_0 = ("Follow structured review system", 0)
generally_on_track_node_0 = ("generally on track", 0)
member_following_role_node_0 = ("member following role", 0)
task_assigned_based_on_role_node_0 = ("task assigned based on role", 0)
reviewer_reviews_task_node_0 = ("reviewer reviews task", 0)
reviewer_provides_feedback_node_0 = ("reviewer provides feedback", 0)
reviewer_reviews_timely_node_0 = ("reviewer reviews timely", 0)
avg_number_of_points_node_0 = ("avg number of points", 0)
num_words_in_task_description_node_0 = ("num words in task description", 0)
task_being_rejected_node_0 = ("task being rejected", 0)
duration_of_task_in_progress_node_0 = ("duration of task in progress", 0)
task_assigned_based_on_timeline_node_0 = ("task assigned based on timeline", 0)
completion_time_follows_timeline_node_0 = ("completion time follows timeline", 0)

agreed_upon_system_node_1 = ("Agreed upon system", 1)
clarity_of_goal_node_1 = ("Clarity of goal", 1)
clarity_of_role_node_1 = ("Clarity of role", 1)
specific_timeline_node_1 = ("Specific timeline", 1)
clarity_of_task_node_1 = ("Clarity of task", 1)
follow_structured_review_system_node_1 = ("Follow structured review system", 1)
generally_on_track_node_1 = ("generally on track", 1)
member_following_role_node_1 = ("member following role", 1)
task_assigned_based_on_role_node_1 = ("task assigned based on role", 1)
reviewer_reviews_task_node_1 = ("reviewer reviews task", 1)
reviewer_provides_feedback_node_1 = ("reviewer provides feedback", 1)
reviewer_reviews_timely_node_1 = ("reviewer reviews timely", 1)
avg_number_of_points_node_1 = ("avg number of points", 1)
num_words_in_task_description_node_1 = ("num words in task description", 1)
task_being_rejected_node_1 = ("task being rejected", 1)
duration_of_task_in_progress_node_1 = ("duration of task in progress", 1)
task_assigned_based_on_timeline_node_1 = ("task assigned based on timeline", 1)
completion_time_follows_timeline_node_1 = ("completion time follows timeline", 1)

MAX_ITERATIONS = 5


def agreed_upon_system_process():
    model = DynamicBayesianNetwork([
        (agreed_upon_system_node_0, clarity_of_goal_node_0),
        (agreed_upon_system_node_0, clarity_of_role_node_0),
        (agreed_upon_system_node_0, follow_structured_review_system_node_0),
        (clarity_of_role_node_0, member_following_role_node_0),
        (clarity_of_role_node_0, task_assigned_based_on_role_node_0),
        (clarity_of_goal_node_0, clarity_of_task_node_0),
        (clarity_of_goal_node_0, generally_on_track_node_0),
        (follow_structured_review_system_node_0, reviewer_reviews_task_node_0),
        (follow_structured_review_system_node_0, reviewer_provides_feedback_node_0),
        (follow_structured_review_system_node_0, reviewer_reviews_timely_node_0),
        (clarity_of_task_node_0, avg_number_of_points_node_0),
        (clarity_of_task_node_0, num_words_in_task_description_node_0),
        (clarity_of_task_node_0, task_being_rejected_node_0),
        (clarity_of_task_node_0, duration_of_task_in_progress_node_0),
        (clarity_of_task_node_0, specific_timeline_node_0),
        (specific_timeline_node_0, completion_time_follows_timeline_node_0),
        (specific_timeline_node_0, task_assigned_based_on_timeline_node_0),

        (agreed_upon_system_node_1, clarity_of_goal_node_1),
        (agreed_upon_system_node_1, clarity_of_role_node_1),
        (agreed_upon_system_node_1, follow_structured_review_system_node_1),
        (clarity_of_role_node_1, member_following_role_node_1),
        (clarity_of_role_node_1, task_assigned_based_on_role_node_1),
        (clarity_of_goal_node_1, clarity_of_task_node_1),
        (clarity_of_goal_node_1, generally_on_track_node_1),
        (follow_structured_review_system_node_1, reviewer_reviews_task_node_1),
        (follow_structured_review_system_node_1, reviewer_provides_feedback_node_1),
        (follow_structured_review_system_node_1, reviewer_reviews_timely_node_1),
        (clarity_of_task_node_1, avg_number_of_points_node_1),
        (clarity_of_task_node_1, num_words_in_task_description_node_1),
        (clarity_of_task_node_1, task_being_rejected_node_1),
        (clarity_of_task_node_1, duration_of_task_in_progress_node_1),
        (clarity_of_task_node_1, specific_timeline_node_1),
        (specific_timeline_node_1, completion_time_follows_timeline_node_1),
        (specific_timeline_node_1, task_assigned_based_on_timeline_node_1),
        (agreed_upon_system_node_0, agreed_upon_system_node_1),
        (clarity_of_role_node_0, clarity_of_role_node_1),
        (clarity_of_goal_node_0, clarity_of_goal_node_1)])

    cpd_agreed_upon_system = TabularCPD(agreed_upon_system_node_0,
                                        variable_card=3, values=[
            [0.2],  # low
            [0.6],  # mid
            [0.2]  # high
        ])

    cpd_clarity_of_goal = TabularCPD(variable=clarity_of_goal_node_0,
                                     variable_card=3,
                                     values=[[0.1, 0.2, 0.7],
                                             [0.4, 0.6, 0.2],
                                             [0.5, 0.2, 0.1]],
                                     evidence=[agreed_upon_system_node_0],
                                     evidence_card=[3])

    cpd_clarity_of_role = TabularCPD(variable=clarity_of_role_node_0,
                                     variable_card=3,
                                     values=[[0.1, 0.2, 0.7],
                                             [0.4, 0.6, 0.2],
                                             [0.5, 0.2, 0.1]],
                                     evidence=[agreed_upon_system_node_0],
                                     evidence_card=[3])

    cpd_follow_structured_review_system = TabularCPD(variable=follow_structured_review_system_node_0,
                                                     variable_card=3,
                                                     values=[[0.1, 0.2, 0.7],
                                                             [0.4, 0.6, 0.2],
                                                             [0.5, 0.2, 0.1]],
                                                     evidence=[agreed_upon_system_node_0],
                                                     evidence_card=[3])

    cpd_generally_on_track = TabularCPD(variable=generally_on_track_node_0, variable_card=2,
                                        values=[[0.8, 0.6, 0.2],
                                                [0.2, 0.4, 0.8]],
                                        evidence=[clarity_of_goal_node_0],
                                        evidence_card=[3])

    cpd_clarity_of_task = TabularCPD(variable=clarity_of_task_node_0,
                                     variable_card=3,
                                     values=[[0.2, 0.4, 0.4],
                                             [0.4, 0.4, 0.55],
                                             [0.4, 0.2, 0.05]],
                                     evidence=[clarity_of_goal_node_0],
                                     evidence_card=[3])

    cpd_member_following_roles = TabularCPD(variable=member_following_role_node_0,
                                            variable_card=3,
                                            values=[[0.8, 0.1, 0.01],
                                                    [0.15, 0.7, 0.3],
                                                    [0.05, 0.2, 0.69]],
                                            evidence=[clarity_of_role_node_0],
                                            evidence_card=[3])

    cpd_task_assigned_by_role = TabularCPD(variable=task_assigned_based_on_role_node_0,
                                           variable_card=3,
                                           values=[[0.8, 0.2, 0.01],
                                                   [0.19, 0.7, 0.3],
                                                   [0.01, 0.1, 0.69]],
                                           evidence=[clarity_of_role_node_0],
                                           evidence_card=[3])

    cpd_reviewer_reviews_task = TabularCPD(variable=reviewer_reviews_task_node_0, variable_card=3,
                                           values=[[0.8, 0.3, 0.05],
                                                   [0.15, 0.55, 0.4],
                                                   [0.05, 0.15, 0.55]],
                                           evidence=[follow_structured_review_system_node_0],
                                           evidence_card=[3])

    cpd_reviewer_provides_feedback = TabularCPD(variable=reviewer_provides_feedback_node_0, variable_card=3,
                                                values=[[0.2, 0.1, 0.01],
                                                        [0.75, 0.6, 0.3],
                                                        [0.05, 0.3, 0.69]],
                                                evidence=[follow_structured_review_system_node_0],
                                                evidence_card=[3])

    cpd_reviewer_reviews_timely = TabularCPD(variable=reviewer_reviews_timely_node_0, variable_card=2,
                                             values=[[0.8, 0.6, 0.1],
                                                     [0.2, 0.4, 0.9]],
                                             evidence=[follow_structured_review_system_node_0],
                                             evidence_card=[3])

    cpd_avg_num_points = TabularCPD(variable=avg_number_of_points_node_0,
                                    variable_card=3,
                                    values=[[0.1, 0.2, 0.7],
                                            [0.25, 0.6, 0.2],
                                            [0.65, 0.2, 0.1]],
                                    evidence=[clarity_of_task_node_0],
                                    evidence_card=[3])

    cpd_num_words_in_description = TabularCPD(variable=num_words_in_task_description_node_0,
                                              variable_card=3,
                                              values=[[0.7, 0.4, 0.1],
                                                      [0.2, 0.4, 0.3],
                                                      [0.1, 0.2, 0.6]],
                                              evidence=[clarity_of_task_node_0],
                                              evidence_card=[3])

    cpd_task_being_rejected = TabularCPD(variable=task_being_rejected_node_0, variable_card=2,
                                         values=[[0.2, 0.3, 0.5],
                                                 [0.8, 0.7, 0.5]],
                                         evidence=[clarity_of_task_node_0],
                                         evidence_card=[3])

    cpd_duration_in_progress = TabularCPD(variable=duration_of_task_in_progress_node_0,
                                          variable_card=3,
                                          values=[[0.2, 0.3, 0.5],
                                                  [0.3, 0.5, 0.4],
                                                  [0.5, 0.2, 0.1]],
                                          evidence=[clarity_of_task_node_0],
                                          evidence_card=[3])

    cpd_specific_timeline = TabularCPD(variable=specific_timeline_node_0, variable_card=2,
                                       values=[[0.9, 0.7, 0.2],
                                               [0.1, 0.3, 0.8]],
                                       evidence=[clarity_of_task_node_0],
                                       evidence_card=[3])

    cpd_task_assigned_based_on_timeline = TabularCPD(variable=task_assigned_based_on_timeline_node_0, variable_card=2,
                                                     values=[[0.9, 0.01],
                                                             [0.1, 0.99]],
                                                     evidence=[specific_timeline_node_0],
                                                     evidence_card=[2])

    cpd_completion_time_follows_timeline = TabularCPD(variable=completion_time_follows_timeline_node_0, variable_card=2,
                                                      values=[[0.95, 0.01],
                                                              [0.05, 0.99]],
                                                      evidence=[specific_timeline_node_0],
                                                      evidence_card=[2])

    cpd_agreed_upon_system_transitional = TabularCPD(variable=agreed_upon_system_node_1, variable_card=3,
                                                     values=[[0.8, 0.15, 0.05],
                                                             [0.15, 0.8, 0.15],
                                                             [0.05, 0.05, 0.8]],
                                                     evidence=[agreed_upon_system_node_0],
                                                     evidence_card=[3])

    cpd_clarity_of_role_transitional = TabularCPD(variable=clarity_of_role_node_1, variable_card=3,
                                                  values=[[0.05, 0.15, 0.8, 0.05, 0.25, 0.7, 0.1, 0.25, 0.65],
                                                          [0.15, 0.8, 0.15, 0.25, 0.7, 0.25, 0.25, 0.65, 0.25],
                                                          [0.8, 0.05, 0.05, 0.7, 0.05, 0.05, 0.65, 0.1, 0.1]],
                                                  evidence=[agreed_upon_system_node_1, clarity_of_role_node_0],
                                                  evidence_card=[3, 3])

    cpd_clarity_of_goal_transitional = TabularCPD(variable=clarity_of_goal_node_1, variable_card=3,
                                                  values=[[0.05, 0.15, 0.8, 0.05, 0.25, 0.7, 0.1, 0.25, 0.65],
                                                          [0.15, 0.8, 0.15, 0.25, 0.7, 0.25, 0.25, 0.65, 0.25],
                                                          [0.8, 0.05, 0.05, 0.7, 0.05, 0.05, 0.65, 0.1, 0.1]],
                                                  evidence=[agreed_upon_system_node_1, clarity_of_goal_node_0],
                                                  evidence_card=[3, 3])

    model.add_cpds(cpd_agreed_upon_system, cpd_clarity_of_goal, cpd_clarity_of_role,
                   cpd_specific_timeline, cpd_clarity_of_task, cpd_follow_structured_review_system,
                   cpd_member_following_roles, cpd_task_assigned_by_role,
                   cpd_num_words_in_description, cpd_generally_on_track,
                   cpd_reviewer_reviews_task, cpd_reviewer_provides_feedback,
                   cpd_reviewer_reviews_timely, cpd_avg_num_points,
                   cpd_task_being_rejected, cpd_duration_in_progress,
                   cpd_task_assigned_based_on_timeline, cpd_completion_time_follows_timeline,
                   cpd_agreed_upon_system_transitional,
                   cpd_clarity_of_role_transitional,
                   cpd_clarity_of_goal_transitional
                   )

    model.initialize_initial_state()
    return model


def group_A_parallel(model):
    # student_1_result_high = pd.DataFrame(index=range(0, MAX_ITERATIONS))
    # student_1_result_mid = pd.DataFrame(index=range(0, MAX_ITERATIONS))
    # student_1_result_low = pd.DataFrame(index=range(0, MAX_ITERATIONS))

    list_of_tuple = []
    with Pool(5) as p:
        list_of_tuple = p.map(group_A_inference, [model] * MAX_ITERATIONS)

    df = pd.DataFrame(list_of_tuple, columns=[1, 2, 3])
    print(df)
    df.to_csv("RESULT.csv")


def group_A_inference(model):
    dbn_inf = DBNInference(model)
    '''
    Number of words in task description = high (0), every week 
    Task assigned based on role = all (0), every 3 weeks
    Completion of task follow timeline = true (0), every 2 weeks
    Reviewer provides meaningful feedback = all(0), every 2 weeks
    '''

    print("week 1")
    # it uses rejection sampling from joint distribution of the bayesian network provided
    sim_1 = model.simulate(n_samples=1, n_time_slices=2, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0
    })
    sim_dict_1 = sim_1.to_dict('records')[0]
    del sim_dict_1[agreed_upon_system_node_1]
    inference_value_1 = dbn_inf.forward_inference([agreed_upon_system_node_1], sim_dict_1)
    result_1 = inference_value_1[agreed_upon_system_node_1].values

    print("week 2")
    sim_2 = model.simulate(n_samples=1, n_time_slices=3, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,

    })
    sim_dict_2 = sim_2.to_dict('records')[0]
    del sim_dict_2[("Agreed upon system", 2)]
    inference_value_2 = dbn_inf.forward_inference([("Agreed upon system", 2)], sim_dict_2)
    result_2 = inference_value_2[("Agreed upon system", 2)].values

    print("week 3")
    sim_3 = model.simulate(n_samples=1, n_time_slices=4, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
    })
    sim_dict_3 = sim_3.to_dict('records')[0]
    del sim_dict_3[("Agreed upon system", 3)]
    inference_value_3 = dbn_inf.forward_inference([("Agreed upon system", 3)], sim_dict_3)
    result_3 = inference_value_3[("Agreed upon system", 3)].values

    print("week 4")
    sim_4 = model.simulate(n_samples=1, n_time_slices=5, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
    })
    sim_dict_4 = sim_4.to_dict('records')[0]
    del sim_dict_4[("Agreed upon system", 4)]
    inference_value_4 = dbn_inf.forward_inference([("Agreed upon system", 4)], sim_dict_4)
    result_4 = inference_value_4[("Agreed upon system", 4)].values

    print("week 5")
    sim_5 = model.simulate(n_samples=1, n_time_slices=6, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
    })
    sim_dict_5 = sim_5.to_dict('records')[0]
    del sim_dict_5[("Agreed upon system", 5)]
    inference_value_5 = dbn_inf.forward_inference([("Agreed upon system", 5)], sim_dict_5)
    result_5 = inference_value_5[("Agreed upon system", 5)].values

    print("week 6")
    sim_6 = model.simulate(n_samples=1, n_time_slices=7, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
        ("num words in task description", 6): 0,
        ("completion time follows timeline", 6): 0,
        ("reviewer provides feedback", 6): 0,
    })
    sim_dict_6 = sim_6.to_dict('records')[0]
    del sim_dict_6[("Agreed upon system", 6)]
    inference_value_6 = dbn_inf.forward_inference([("Agreed upon system", 6)], sim_dict_6)
    result_6 = inference_value_6[("Agreed upon system", 6)].values

    print("week 7")
    sim_7 = model.simulate(n_samples=1, n_time_slices=8, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
        ("num words in task description", 6): 0,
        ("completion time follows timeline", 6): 0,
        ("reviewer provides feedback", 6): 0,
        ("num words in task description", 7): 0,
        ("task assigned based on role", 7): 0,
    })
    sim_dict_7 = sim_7.to_dict('records')[0]
    del sim_dict_7[("Agreed upon system", 7)]
    inference_value_7 = dbn_inf.forward_inference([("Agreed upon system", 7)], sim_dict_7)
    result_7 = inference_value_7[("Agreed upon system", 7)].values

    print("week 8")
    sim_8 = model.simulate(n_samples=1, n_time_slices=9, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
        ("num words in task description", 6): 0,
        ("completion time follows timeline", 6): 0,
        ("reviewer provides feedback", 6): 0,
        ("num words in task description", 7): 0,
        ("task assigned based on role", 7): 0,
        ("num words in task description", 8): 0,
        ("completion time follows timeline", 8): 0,
        ("reviewer provides feedback", 8): 0,
    })
    sim_dict_8 = sim_8.to_dict('records')[0]
    del sim_dict_8[("Agreed upon system", 8)]
    inference_value_8 = dbn_inf.forward_inference([("Agreed upon system", 8)], sim_dict_8)
    result_8 = inference_value_8[("Agreed upon system", 8)].values

    print("week 9")
    sim_9 = model.simulate(n_samples=1, n_time_slices=10, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
        ("num words in task description", 6): 0,
        ("completion time follows timeline", 6): 0,
        ("reviewer provides feedback", 6): 0,
        ("num words in task description", 7): 0,
        ("task assigned based on role", 7): 0,
        ("num words in task description", 8): 0,
        ("completion time follows timeline", 8): 0,
        ("reviewer provides feedback", 8): 0,
        ("num words in task description", 9): 0,
    })
    sim_dict_9 = sim_9.to_dict('records')[0]
    del sim_dict_9[("Agreed upon system", 9)]
    inference_value_9 = dbn_inf.forward_inference([("Agreed upon system", 9)], sim_dict_9)
    result_9 = inference_value_9[("Agreed upon system", 9)].values

    print("week 10")
    sim_10 = model.simulate(n_samples=1, n_time_slices=11, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
        ("num words in task description", 6): 0,
        ("completion time follows timeline", 6): 0,
        ("reviewer provides feedback", 6): 0,
        ("num words in task description", 7): 0,
        ("task assigned based on role", 7): 0,
        ("num words in task description", 8): 0,
        ("completion time follows timeline", 8): 0,
        ("reviewer provides feedback", 8): 0,
        ("num words in task description", 9): 0,
        ("task assigned based on role", 10): 0,
        ("num words in task description", 10): 0,
        ("completion time follows timeline", 10): 0,
        ("reviewer provides feedback", 10): 0,
    })
    sim_dict_10 = sim_10.to_dict('records')[0]
    del sim_dict_10[("Agreed upon system", 10)]
    inference_value_10 = dbn_inf.forward_inference([("Agreed upon system", 10)], sim_dict_10)
    result_10 = inference_value_10[("Agreed upon system", 10)].values

    print("week 11")
    sim_11 = model.simulate(n_samples=1, n_time_slices=12, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
        ("num words in task description", 6): 0,
        ("completion time follows timeline", 6): 0,
        ("reviewer provides feedback", 6): 0,
        ("num words in task description", 7): 0,
        ("task assigned based on role", 7): 0,
        ("num words in task description", 8): 0,
        ("completion time follows timeline", 8): 0,
        ("reviewer provides feedback", 8): 0,
        ("num words in task description", 9): 0,
        ("task assigned based on role", 10): 0,
        ("num words in task description", 10): 0,
        ("completion time follows timeline", 10): 0,
        ("reviewer provides feedback", 10): 0,
        ("num words in task description", 11): 0,
    })
    sim_dict_11 = sim_11.to_dict('records')[0]
    del sim_dict_11[("Agreed upon system", 11)]
    inference_value_11 = dbn_inf.forward_inference([("Agreed upon system", 11)], sim_dict_11)
    result_11 = inference_value_11[("Agreed upon system", 11)].values

    print("week 12")
    sim_12 = model.simulate(n_samples=1, n_time_slices=13, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
        ("num words in task description", 6): 0,
        ("completion time follows timeline", 6): 0,
        ("reviewer provides feedback", 6): 0,
        ("num words in task description", 7): 0,
        ("task assigned based on role", 7): 0,
        ("num words in task description", 8): 0,
        ("completion time follows timeline", 8): 0,
        ("reviewer provides feedback", 8): 0,
        ("num words in task description", 9): 0,
        ("task assigned based on role", 10): 0,
        ("num words in task description", 10): 0,
        ("completion time follows timeline", 10): 0,
        ("reviewer provides feedback", 10): 0,
        ("num words in task description", 11): 0,
        ("num words in task description", 12): 0,
        ("completion time follows timeline", 12): 0,
        ("reviewer provides feedback", 12): 0,
    })
    sim_dict_12 = sim_12.to_dict('records')[0]
    del sim_dict_12[("Agreed upon system", 12)]
    inference_value_12 = dbn_inf.forward_inference([("Agreed upon system", 12)], sim_dict_12)
    result_12 = inference_value_12[("Agreed upon system", 12)].values

    print("week 13")
    sim_13 = model.simulate(n_samples=1, n_time_slices=14, evidence={
        ("task assigned based on role", 1): 0,
        ("num words in task description", 1): 0,
        ("num words in task description", 2): 0,
        ("completion time follows timeline", 2): 0,
        ("reviewer provides feedback", 2): 0,
        ("num words in task description", 3): 0,
        ("task assigned based on role", 4): 0,
        ("num words in task description", 4): 0,
        ("completion time follows timeline", 4): 0,
        ("reviewer provides feedback", 4): 0,
        ("num words in task description", 5): 0,
        ("num words in task description", 6): 0,
        ("completion time follows timeline", 6): 0,
        ("reviewer provides feedback", 6): 0,
        ("num words in task description", 7): 0,
        ("task assigned based on role", 7): 0,
        ("num words in task description", 8): 0,
        ("completion time follows timeline", 8): 0,
        ("reviewer provides feedback", 8): 0,
        ("num words in task description", 9): 0,
        ("task assigned based on role", 10): 0,
        ("num words in task description", 10): 0,
        ("completion time follows timeline", 10): 0,
        ("reviewer provides feedback", 10): 0,
        ("num words in task description", 11): 0,
        ("num words in task description", 12): 0,
        ("completion time follows timeline", 12): 0,
        ("reviewer provides feedback", 12): 0,
        ("num words in task description", 13): 0,
    })
    sim_dict_13 = sim_13.to_dict('records')[0]
    del sim_dict_13[("Agreed upon system", 13)]
    inference_value_13 = dbn_inf.forward_inference([("Agreed upon system", 13)], sim_dict_13)
    result_13 = inference_value_13[("Agreed upon system", 13)].values
    return (
    result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11,
    result_12, result_13)


if __name__ == '__main__':
    model = agreed_upon_system_process()
    group_A_parallel(model)
