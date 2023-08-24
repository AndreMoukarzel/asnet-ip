from typing import List
import json

from .training_helper import TrainingHelper

import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping


class Trainer:

    def __init__(self, domain_file: str, problem_files: List[str], validation_problem_file: str=None) -> None:
        self.helpers: List[TrainingHelper] = []
        for prob_file in problem_files:
            helper = TrainingHelper(domain_file, prob_file) 
            helper.net.compile()
            self.helpers.append(helper)
        

        if validation_problem_file:
            self.val_helper = TrainingHelper(domain_file, validation_problem_file)
            self.val_helper.net.compile()
    

    def _check_planning_success(self, shared_weights) -> bool:
        # If a validation problem is defined, getting to the goal of the
        # validation problem is enough
        if hasattr(self, 'val_helper'):
            self.val_helper.set_model_weights(shared_weights)
            states, _ = self.val_helper.run_policy(self.val_helper.init_state)
            if self.val_helper.is_goal(states[-1]):
                return True
            return False
        
        # If no validation problem is defined, we check if the weights are
        # capable of reaching the goal of all training problems
        for helper in self.helpers:
            helper.set_model_weights(shared_weights)
            states, _ = helper.run_policy(helper.init_state)
            if not helper.is_goal(states[-1]):
                return False
        return True
    

    def train(self, full_epochs: int = 500, train_epochs: int = 100, verbose: int = 0) -> list:
        """
        """
        # Configures Early Stopping configuration for training
        callback = EarlyStopping(monitor='auc', patience=20, min_delta=0.001)

        shared_weights = self.helpers[0].get_model_weights()

        consecutive_solved: int = 0 # Number the problems were successfully solved consecutively
        histories: list = [[]] * len(self.helpers)
        try:
            for _ in tqdm(range(full_epochs)):
                for i, helper in enumerate(self.helpers):
                    converted_states, converted_actions = helper.generate_training_inputs(verbose=verbose)
                    minibatch_size: int = len(converted_states)//2 # Hard-coded batch size to be half of training set

                    helper.set_model_weights(shared_weights) # Overwrite model with the weights being trained
                    model = helper.net.get_model()
                    history = model.fit(converted_states, converted_actions, epochs=train_epochs, batch_size=minibatch_size, callbacks=[callback], verbose=verbose)
                    histories[i].append(history)
                    shared_weights = helper.get_model_weights()

                # Custom Early Stopping
                if self._check_planning_success(shared_weights):
                    consecutive_solved += 1
                    if consecutive_solved >= 20:
                        break
                else:
                    consecutive_solved = 0
        except KeyboardInterrupt:
            pass
        return histories, shared_weights


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    domain = 'problems/deterministic_blocksworld/domain.pddl'
    problems = [
        'problems/deterministic_blocksworld/pb5_p00.pddl',
        'problems/deterministic_blocksworld/pb5_p01.pddl',
        'problems/deterministic_blocksworld/pb5_p02.pddl'
    ]
    val_problem = 'problems/deterministic_blocksworld/pb5_p03.pddl'

    print("Instancing ASNets")
    trainer = Trainer(domain, problems, val_problem)
    print("Training")
    _, weights = trainer.train(verbose=1)

    with open('data/new_output.json', 'w') as f:
        json.dump(weights, f, cls=NumpyEncoder)
