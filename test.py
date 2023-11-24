import json

from asnet.training_helper import TrainingHelper

import numpy as np



with open('data/deterministic_blocksworld.json', 'r') as f:
    weights: dict = json.load(f)
    for key, value in weights.items():
        output_weights = np.array([np.array(val) for val in value[0]])
        bias = np.array(value[1])
        weights[key] = (output_weights, bias)

helper = TrainingHelper(
        'problems/deterministic_blocksworld/domain.pddl',
        'problems/deterministic_blocksworld/pb5_p01.pddl',
        solve=False
    )

"""
helper = TrainingHelper(
        'problems/deterministic_blocksworld/domain.pddl',
        'problems/deterministic_blocksworld/pb3.pddl',
        solve=False
    )
"""

helper.set_model_weights(weights)

print("Executing a test policy with the Network")
states, actions = helper.run_policy(helper.init_state)
for i, act in enumerate(actions):
    print("State: ", states[i])
    print("Action taken: ", act)
print("State: ", states[-1])
if helper.is_goal(states[-1]):
    print("GOAL REACHED")
