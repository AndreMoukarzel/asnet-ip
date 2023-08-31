from typing import List, Dict
import json

from .training_helper import TrainingHelper

import click
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping


class Trainer:
    """
    Class responsible for training an ASNet in a specified domain.

    As such, multiple of the domain's problem instances can be specified, so the
    ASNet may be trained in multiple of them in tandem so it may learn
    generalized weights capable of solving multiple problem instances in the
    same domain.

    Methods
    -------
    train(full_epochs, train_epochs, verbose)
        Trains the ASNet, returning both the training history and the ASNet's
        trained weights.
    """

    def __init__(self, domain_file: str, problem_files: List[str], validation_problem_file: str='') -> None:
        """
        Parameters
        ----------
        domain_file : str
            IPPDDL file specifying the problem domain
        problem_files : List[str]
            List of IPPDDL files specifying the problem instances to be used in
            training the ASNet
        validation_problem_file
            IPPDDL file specifying a problem isntace to be used as a validation
            instance when training the ASNet. More details in
            _check_planning_success()
        """
        self.helpers: List[TrainingHelper] = [TrainingHelper(domain_file, problem_files[0])]
        if len(problem_files) > 1:
            for prob_file in problem_files[1:]:
                helper = TrainingHelper(domain_file, prob_file, instance_asnet=False) 
                self.helpers.append(helper)
        
        if validation_problem_file != '':
            self.val_helper = TrainingHelper(domain_file, validation_problem_file, instance_asnet=False)
    

    def _check_planning_success(self, model):#shared_weights: Dict[str, np.array]) -> bool:
        """Returns if the ASNet's planning was successful.
        
        If a validation problem is defined, returns true if the ASNet sucessfully
        solved de validation problem. Otherwise, is sucessful if it solves all
        training problems.
        
        Parameters
        ----------
        shared_weights: Dict[str, np.array]
            Generalized weights of an ASNet.
        
        Returns
        -------
        bool
            True if the planning was successful, false otherwise.
        """
        # If a validation problem is defined, getting to the goal of the
        # validation problem is enough
        if hasattr(self, 'val_helper'):
            #self.val_helper.set_model_weights(shared_weights)
            states, _ = self.val_helper.run_policy(self.val_helper.init_state, model)
            if self.val_helper.is_goal(states[-1]):
                return True
            return False
        
        # If no validation problem is defined, we check if the weights are
        # capable of reaching the goal of all training problems
        for helper in self.helpers:
            #helper.set_model_weights(shared_weights)
            states, _ = helper.run_policy(helper.init_state, model)
            if not helper.is_goal(states[-1]):
                return False
        return True
    

    def train(self, exploration_loops: int = 550, train_epochs: int = 100, verbose: int = 0) -> list:
        """Trains an ASNet in the defined problem instances.

        Training will be stopped early if _check_planning_success() returns true
        in 20 consecutive executions.

        Parameters
        ----------
        exploration_loops: int, optional
            Maximum number training inputs will be generated.
        train_epochs: int, optional
            Number of training epochs for each exploration loop, and therefore
            the same training set.
        verbose: int, optional
            Verbosity mode when running the ASNet's policy and training.
            0 = silent, 1 = progress bar, 2 = one line per epoch.

        Returns
        -------
        list
            Returns both the training history, where each element is the history
            of an exploration loop, as well as the final shared weights from
            the model.
        """
        # Configures Early Stopping configuration for training
        callback = EarlyStopping(monitor='loss', patience=20, min_delta=0.001)

        model = self.helpers[0].get_model()

        consecutive_solved: int = 0 # Number the problems were successfully solved consecutively
        histories: list = [[]] * len(self.helpers)
        try:
            for _ in tqdm(range(exploration_loops)):
                for i, helper in enumerate(self.helpers):
                    converted_states, converted_actions = helper.generate_training_inputs(model, verbose=verbose)
                    minibatch_size: int = len(converted_states)//2 # Hard-coded batch size to be half of training set

                    history = model.fit(converted_states, converted_actions, epochs=train_epochs, batch_size=minibatch_size, callbacks=[callback], verbose=verbose)
                    histories[i].append(history)

                # Custom Early Stopping
                if self._check_planning_success(model):
                    consecutive_solved += 1
                    if consecutive_solved >= 20:
                        print(f"Reached goal in {20} consecutive iterations.")
                        break
                else:
                    consecutive_solved = 0
        except KeyboardInterrupt:
            pass

        shared_weights = self.helpers[0].get_model_weights()
        return histories, shared_weights


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@click.command()
@click.option("--domain", "-d", type=str, help="Path to the problem's domain PPDDL file.", default='problems/deterministic_blocksworld/domain.pddl')
@click.option(
    "--problems", "-p", type=str, help="Path to (multiple) problem's instance PPDDL files.", multiple=True,
    default=[
        'problems/deterministic_blocksworld/pb5_p00.pddl',
        'problems/deterministic_blocksworld/pb5_p01.pddl',
        'problems/deterministic_blocksworld/pb5_p02.pddl'
    ])
@click.option("--valid", "-v", type=str, help="Path to problem's instance PPDDL files used for training's validation.", default='')
@click.option("--save", "-s", type=str, help="Name of file with saved weights after training. If none is set, used the domain's name.", default='')
@click.option("--verbose", type=int, help="Debug prints. Off by default.", default=0)
def execute(domain, problems, valid: str, save: str, verbose: int):
    print("Instancing ASNets")
    trainer = Trainer(domain, problems, valid)
    print("Training")
    _, weights = trainer.train(verbose=verbose)

    if save == '':
        save = domain.split('/')[-2]
    with open(f'data/{save}.json', 'w') as f:
        json.dump(weights, f, cls=NumpyEncoder)



if __name__ == "__main__":
    execute()
