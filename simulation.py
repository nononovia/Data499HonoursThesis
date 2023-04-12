from pgmpy.inference import DBNInference
from Commitment_DBN import commitment



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
review_work_from_oters_node_1 = ('Review work from others', 1)
initiate_meeting_node_1 = ('Initiate meeting', 1)

def run_simulation():
    model = commitment()
    dbn_inf = DBNInference(model)

    print('User 1')
    _user_one(dbn_inf)
    # _display_results(user_one_results)


def _user_one(dbn_inf):
    """
    Solid student who does the basic jobs but is not enthusiastic
    """

    probabilities = []
    inference_value = dbn_inf.forward_inference([commitment_node_1], {
        task_done_on_time_node_1: 1,
        positive_tone_node_1: 0,
    })
    print(inference_value[commitment_node_1].values)
    inference_value_1 = dbn_inf.forward_inference([commitment_node_1], {
        commitment_node_0: 0,
        task_done_on_time_node_1: 1,
        positive_tone_node_1: 0,
    })
    print(inference_value_1[commitment_node_1].values)

    copy_model = model.copy()


if __name__ == '__main__':
    run_simulation()

