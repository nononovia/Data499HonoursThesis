import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference

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

MAX_ITERATIONS = 10
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
                                                  values=[[0.8, 0.15, 0.05, 0.7, 0.25, 0.05, 0.65, 0.25, 0.1],
                                                              [0.15, 0.8, 0.15, 0.25, 0.7, 0.25, 0.25, 0.65, 0.25],
                                                              [0.05, 0.05, 0.8, 0.05, 0.05, 0.7, 0.1, 0.1, 0.658]],
                                                  evidence=[clarity_of_role_node_0, agreed_upon_system_node_1],
                                                  evidence_card=[3, 3])

    cpd_clarity_of_goal_transitional = TabularCPD(variable=clarity_of_goal_node_1, variable_card=3,
                                                  values=[[0.8, 0.15, 0.05, 0.7, 0.25, 0.05, 0.65, 0.25, 0.1],
                                                          [0.15, 0.8, 0.15, 0.25, 0.7, 0.25, 0.25, 0.65, 0.25],
                                                          [0.05, 0.05, 0.8, 0.05, 0.05, 0.7, 0.1, 0.1, 0.658]],
                                                  evidence=[clarity_of_goal_node_0, agreed_upon_system_node_1],
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


def group_A(model):
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    week1_result = []
    week2_result = []
    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        student_1_result_high.iloc[i, 0] = 0.2
        student_1_result_mid.iloc[i, 0] = 0.6
        student_1_result_low.iloc[i, 0] = 0.2

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            task_assigned_based_on_role_node_1: 0,
            avg_number_of_points_node_1: 2,
            task_assigned_based_on_timeline_node_1: 0
        })
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[agreed_upon_system_node_1]
        inference_value = dbn_inf.forward_inference([agreed_upon_system_node_1], sim_dict)
        result = inference_value[agreed_upon_system_node_1].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week1_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            task_assigned_based_on_role_node_1: 0,
            avg_number_of_points_node_1: 2,
            task_assigned_based_on_timeline_node_1: 0,
            ("member following role", 2): 0,
            ("num words in task description", 2): 0,
            ("reviewer reviews task", 2): 0
        })
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Agreed upon system", 2)]
        inference_value = dbn_inf.forward_inference([("Agreed upon system", 2)], sim_dict)
        result = inference_value[("Agreed upon system", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            task_assigned_based_on_role_node_1: 0,
            avg_number_of_points_node_1: 2,
            task_assigned_based_on_timeline_node_1: 0,
            ("member following role", 2): 0,
            ("num words in task description", 2): 0,
            ("reviewer reviews task", 2): 0,
            ("member following role", 3): 0,
            ("completion time follows timeline", 3): 0,
            ("reviewer provides feedback", 3): 0
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Agreed upon system", 3)]
        inference_value = dbn_inf.forward_inference([("Agreed upon system", 3)], sim_dict)
        result = inference_value[("Agreed upon system", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        student_1_result_high.to_pickle("Group_A_high.p")
        student_1_result_mid.to_pickle("Group_A_mid.p")
        student_1_result_low.to_pickle("Group_A_low.p")


def group_B(model):
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    week1_result = []
    week2_result = []
    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        student_1_result_high.iloc[i, 0] = 0.2
        student_1_result_mid.iloc[i, 0] = 0.6
        student_1_result_low.iloc[i, 0] = 0.2

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            task_assigned_based_on_role_node_1: 1,
            num_words_in_task_description_node_1: 1,
        })
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[agreed_upon_system_node_1]
        inference_value = dbn_inf.forward_inference([agreed_upon_system_node_1], sim_dict)
        result = inference_value[agreed_upon_system_node_1].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week1_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            task_assigned_based_on_role_node_1: 1,
            num_words_in_task_description_node_1: 1,
            ("member following role", 2): 1,
            ("duration of task in progress", 2): 1,
            ("reviewer provides feedback", 2): 1,
        })
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Agreed upon system", 2)]
        inference_value = dbn_inf.forward_inference([("Agreed upon system", 2)], sim_dict)
        result = inference_value[("Agreed upon system", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            task_assigned_based_on_role_node_1: 1,
            num_words_in_task_description_node_1: 1,
            ("member following role", 2): 1,
            ("duration of task in progress", 2): 1,
            ("reviewer provides feedback", 2): 1,
            ("generally on track", 3): 0,
            ("avg number of points", 3): 1
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Agreed upon system", 3)]
        inference_value = dbn_inf.forward_inference([("Agreed upon system", 3)], sim_dict)
        result = inference_value[("Agreed upon system", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        student_1_result_high.to_pickle("Group_B_high_2.p")
        student_1_result_mid.to_pickle("Group_B_mid_2.p")
        student_1_result_low.to_pickle("Group_B_low_2.p")


def group_C(model):
    dbn_inf = DBNInference(model)

    student_1_result_high = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    student_1_result_mid = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    student_1_result_low = pd.DataFrame(columns=range(0, 4), index=range(0, MAX_ITERATIONS))
    week1_result = []
    week2_result = []
    for i in range(0, MAX_ITERATIONS):
        print("week 0")
        student_1_result_high.iloc[i, 0] = 0.2
        student_1_result_mid.iloc[i, 0] = 0.6
        student_1_result_low.iloc[i, 0] = 0.2

        print("week 1")
        # it uses rejection sampling from joint distribution of the bayesian network provided
        sim = model.simulate(n_samples=1, n_time_slices=2, evidence={
            task_assigned_based_on_role_node_1: 2,
            num_words_in_task_description_node_1: 2,
        })
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[agreed_upon_system_node_1]
        inference_value = dbn_inf.forward_inference([agreed_upon_system_node_1], sim_dict)
        result = inference_value[agreed_upon_system_node_1].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week1_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 1] = commitment_high_prob
        student_1_result_mid.iloc[i, 1] = commitment_mid_prob
        student_1_result_low.iloc[i, 1] = commitment_low_prob

        print("week 2")
        sim = model.simulate(n_samples=1, n_time_slices=3, evidence={
            task_assigned_based_on_role_node_1: 2,
            num_words_in_task_description_node_1: 2,
            ("task being rejected", 2): 0,
            ("task assigned based on timeline", 2): 1,
            ("reviewer reviews task", 2): 2,
        })
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Agreed upon system", 2)]
        inference_value = dbn_inf.forward_inference([("Agreed upon system", 2)], sim_dict)
        result = inference_value[("Agreed upon system", 2)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 2] = commitment_high_prob
        student_1_result_mid.iloc[i, 2] = commitment_mid_prob
        student_1_result_low.iloc[i, 2] = commitment_low_prob

        print("week 3")
        sim = model.simulate(n_samples=1, n_time_slices=4, evidence={
            task_assigned_based_on_role_node_1: 2,
            num_words_in_task_description_node_1: 2,
            ("task being rejected", 2): 0,
            ("task assigned based on timeline", 2): 1,
            ("reviewer reviews task", 2): 2,
            ("duration of task in progress", 3): 0,
            ("reviewer provides feedback", 3): 2
        })  # task done on time _all help others deep
        sim_dict = sim.to_dict('records')[0]
        del sim_dict[("Agreed upon system", 3)]
        inference_value = dbn_inf.forward_inference([("Agreed upon system", 3)], sim_dict)
        result = inference_value[("Agreed upon system", 3)].values
        commitment_low_prob = result[0]
        commitment_mid_prob = result[1]
        commitment_high_prob = result[2]
        week2_result.append(commitment_high_prob)
        student_1_result_high.iloc[i, 3] = commitment_high_prob
        student_1_result_mid.iloc[i, 3] = commitment_mid_prob
        student_1_result_low.iloc[i, 3] = commitment_low_prob

        student_1_result_high.to_pickle("Group_C_high_2.p")
        student_1_result_mid.to_pickle("Group_C_mid_2.p")
        student_1_result_low.to_pickle("Group_C_low_2.p")


if __name__ == '__main__':
    model = agreed_upon_system_process()
    # group_A(model)
    group_B(model)
    group_C(model)
