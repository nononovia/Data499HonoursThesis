
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('commitment', 'completes_work'), ('commitment', 'enthusiastic'), ('commitment', 'go_above_beyond'), ('commitment', 'takes_charge'),
                         ('completes_work', 'task_done_on_time'), ('completes_work', 'work_accepted_by_others'),
                         ('enthusiastic', 'positive_tone'), ('enthusiastic', 'initiate_conversation'),
                         ('go_above_beyond', 'help_others'), ('go_above_beyond', 'complete_more_tasks'),
                         ('takes_charge', 'assigns_tasks'), ('takes_charge', 'initiate_meetings'), ('takes_charge', 'review_others_work')])

# Defining individual CPDs.
cpd_commitment = TabularCPD(variable='commitment', variable_card=2, values=[[0.6], [0.4]])


cpd_completes_work = TabularCPD(variable='completes_work', variable_card=2,
                   values=[[0.9, 0.2],
                           [0.1, 0.8]],
                   evidence=['commitment'],
                   evidence_card=[2])

cpd_enthusiastic = TabularCPD(variable='enthusiastic', variable_card=2,
                   values=[[0.9, 0.2],
                           [0.1, 0.8]],
                   evidence=['commitment'],
                   evidence_card=[2])

cpd_go_above_beyond = TabularCPD(variable='go_above_beyond', variable_card=2,
                   values=[[0.9, 0.2],
                           [0.1, 0.8]],
                   evidence=['commitment'],
                   evidence_card=[2])

cpd_takes_charge = TabularCPD(variable='takes_charge', variable_card=2,
                   values=[[0.9, 0.2],
                           [0.1, 0.8]],
                   evidence=['commitment'],
                   evidence_card=[2])

cpd_task_done_on_time = TabularCPD(variable='task_done_on_time', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['completes_work'],
                   evidence_card=[2])

cpd_work_accepted_by_others = TabularCPD(variable='work_accepted_by_others', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['completes_work'],
                   evidence_card=[2])

cpd_positive_tone = TabularCPD(variable='positive_tone', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['enthusiastic'],
                   evidence_card=[2])

cpd_initiate_conversation = TabularCPD(variable='initiate_conversation', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['enthusiastic'],
                   evidence_card=[2])

cpd_help_others = TabularCPD(variable='help_others', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['go_above_beyond'],
                   evidence_card=[2])

cpd_complete_more_tasks = TabularCPD(variable='complete_more_tasks', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['go_above_beyond'],
                   evidence_card=[2])

cpd_assigns_tasks = TabularCPD(variable='assigns_tasks', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['takes_charge'],
                   evidence_card=[2])

cpd_initiate_meetings = TabularCPD(variable='initiate_meetings', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['takes_charge'],
                   evidence_card=[2])

cpd_review_others_work = TabularCPD(variable='review_others_work', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['takes_charge'],
                   evidence_card=[2])

# Associating the CPDs with the network
model.add_cpds(cpd_commitment, cpd_completes_work, cpd_enthusiastic, cpd_go_above_beyond, cpd_takes_charge,
               cpd_task_done_on_time, cpd_work_accepted_by_others, cpd_positive_tone, cpd_initiate_conversation,
               cpd_help_others, cpd_complete_more_tasks, cpd_assigns_tasks, cpd_initiate_meetings, cpd_review_others_work)

# print(model.check_model())
# print(cpd_review_others_work)

infer = VariableElimination(model)
# QUESTION: what are the implications of no observation
print("Scenario A: Solid student who does the basic jobs but is not enthusiastic")
commitment_dist_A = infer.query(['commitment'], evidence={'task_done_on_time': 1, 'positive_tone': 0})
print(commitment_dist_A)
print("Scenario B: does tasks but poorly")
commitment_dist_B = infer.query(['commitment'], evidence={'task_done_on_time': 1, 'work_accepted_by_others': 0, 'initiate_conversation': 0})
print(commitment_dist_B)
print("Scenario C: Does nothing, and negative")
commitment_dist_C = infer.query(['commitment'], evidence={'task_done_on_time': 0, 'positive_tone': 0})
print(commitment_dist_C)
print("Scenario D: proactive project manager")
commitment_dist_D = infer.query(['commitment'], evidence={'task_done_on_time': 1, 'help_others': 1, 'review_others_work': 1})
print(commitment_dist_D)



# simulation setup
