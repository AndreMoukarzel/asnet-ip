import json

from asnet.training_helper import TrainingHelper

import numpy as np


# Consegui atingir o objetivo de múltiplos problemas quando mantive a camada de output como treinável, mas seria possível
# transferir estes pesos de maneira honesta com instancias de tamanhos diferentes?

# Possivelmente o problema é justamente a camada de output ser densa. Talvez o correto seja ser algum tipo de concatenação
# dos outputs da camada anterior. A camada densa provavelmente liga cada output da camada anterior a todas as ações de saída
# possível, tornando impossível, por exemplo, transferir seus pesos para uma ASNet de uma instancia de tamanho diferente

with open('data/custom_layers2.json', 'r') as f:
    weights: dict = json.load(f)
    for key, value in weights.items():
        output_weights = np.array([np.array(val) for val in value[0]])
        bias = np.array(value[1])
        weights[key] = (output_weights, bias)

helper = TrainingHelper(
        'problems/deterministic_blocksworld/domain.pddl',
        'problems/deterministic_blocksworld/pb5_p01.pddl'
    )

"""
helper = TrainingHelper(
        'problems/deterministic_blocksworld/domain.pddl',
        'problems/deterministic_blocksworld/pb3.pddl'
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
