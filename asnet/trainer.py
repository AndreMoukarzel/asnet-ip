from typing import List
import json
import time
import os

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

    def __init__(self, domain_file: str, problem_files: List[str],
                 validation_problem_file: str='', asnet_layers: int=2
                 ) -> None:
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
        self.info: dict = {} # Saves information about training
        self.info['asnet_layers'] = asnet_layers
        self.info['domain'] = domain_file
        tic = time.process_time()
        self.helpers: List[TrainingHelper] = [TrainingHelper(domain_file, problem_files[0], asnet_layers=asnet_layers)]
        self.info['problems'] = [{problem_files[0]: self.helpers[0].info}]
        if len(problem_files) > 1:
            for prob_file in problem_files[1:]:
                helper = TrainingHelper(domain_file, prob_file, asnet_layers=asnet_layers)
                self.helpers.append(helper)
                self.info['problems'].append({prob_file: helper.info})
        
        self.info['validation_problem'] = {}
        if validation_problem_file != '':
            self.val_helper = TrainingHelper(domain_file, validation_problem_file, asnet_layers=asnet_layers)
            self.info['validation_problem'] = {validation_problem_file, self.val_helper.info}
        toc = time.process_time()
        self.info["instantiation_time"] = toc-tic
    

    def _check_planning_success(self) -> bool:
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
            model = self.val_helper.get_model()
            states, _ = self.val_helper.run_policy(self.val_helper.init_state, model)
            if self.val_helper.is_goal(states[-1]):
                return True
            return False
        
        # If no validation problem is defined, we check if the weights are
        # capable of reaching the goal of all training problems
        for helper in self.helpers:
            model = helper.get_model()
            states, _ = helper.run_policy(helper.init_state, model)
            if not helper.is_goal(states[-1]):
                return False
        return True
    

    def update_helpers_weights(self):
        # Assumes the most recently trained helper was the last in list
        weights = self.helpers[-1].get_model_weights() # Get weights of previous helper
        for helper in self.helpers:
            helper.set_model_weights(weights)
        
        if hasattr(self, 'val_helper'):
            self.val_helper.set_model_weights(weights)
    

    def get_starting_iteration(self, save_path: str):
        if not os.path.isdir(save_path):
            return 0 # Save path not found
        files = [f.split('.json')[0] for f in os.listdir(save_path) if '.json' in f]
        if len(files) == 0:
            return 0 # No intermediary saved model found, start from beginning
        iter_values = [int(f.split('iter')[-1]) for f in files]
        print(f'Starting training from iteration {max(iter_values) + 1}')
        return max(iter_values) + 1 # Starts from the next iteration after the saved models'


    def save_intermediary_weights(self, iteration: int, save_path: str):
        if iteration > 0 and iteration % 10 == 0: # Saves intermediary results each 10 iterations of training
            with open(f'{save_path}/iter{iteration}.json', 'w') as f:
                json.dump(self.helpers[-1].get_model_weights(), f, cls=NumpyEncoder)
    

    def train_with_exploration(self, exploration_loops: int = 550, train_epochs: int = 100, verbose: int = 0, save_path: str='/data/models') -> list:
        # Configures Early Stopping configuration for training
        callback = EarlyStopping(monitor='loss', patience=20, min_delta=0.001)

        model = self.helpers[0].get_model()

        self.info['starting_iteration'] = self.get_starting_iteration(save_path)
        if self.info['starting_iteration'] > 0:
            self.helpers[0].set_model_weights_from_file(f"{save_path}/iter{self.info['starting_iteration'] - 1}.json")

        consecutive_solved: int = 0 # Number the problems were successfully solved consecutively
        histories: list = [[]] * len(self.helpers)
        try:
            for iter in tqdm(range(self.get_starting_iteration(save_path), exploration_loops)):
                for i, helper in enumerate(self.helpers):
                    # Updates model with latest trained weights
                    weights = self.helpers[(i - 1) % len(self.helpers)].get_model_weights() # Get weights of previous helper
                    helper.set_model_weights(weights)
                    model = helper.get_model()

                    converted_states, converted_actions = helper.generate_training_inputs(model, self.info['policy_exploration'], verbose=verbose)
                    minibatch_size: int = len(converted_states)//2 # Hard-coded batch size to be half of training set
                    
                    history = model.fit(converted_states, converted_actions, epochs=train_epochs, batch_size=minibatch_size, callbacks=[callback], verbose=verbose)

                    histories[i].append(history)

                self.info["training_iterations"] = iter
                # Custom Early Stopping
                self.update_helpers_weights()
                if self._check_planning_success():
                    consecutive_solved += 1
                    if consecutive_solved >= 20:
                        print(f"Reached goal in {20} consecutive iterations.")
                        self.info["early_solving"] = True
                        break
                else:
                    consecutive_solved = 0
                    self.save_intermediary_weights(iter, save_path)
        except KeyboardInterrupt:
            self.info["early_stopped"] = True
        
        return histories
    

    def train_without_exploration(self, exploration_loops: int = 550, train_epochs: int = 100, verbose: int = 0, save_path: str='/data/models') -> list:
         # Configures Early Stopping configuration for training
        callback = EarlyStopping(monitor='loss', patience=20, min_delta=0.001)

        model = self.helpers[0].get_model()

        consecutive_solved: int = 0 # Number the problems were successfully solved consecutively
        histories: list = [[]] * len(self.helpers)
        training_inputs = []
        for helper in self.helpers:
            converted_states, converted_actions = helper.generate_training_inputs(policy_exploration=self.info['policy_exploration'])
            training_inputs.append((converted_states, converted_actions))
        try:
            for iter in tqdm(range(self.get_starting_iteration(save_path), exploration_loops)):
                for i, helper in enumerate(self.helpers):
                    # Updates model with latest trained weights
                    weights = self.helpers[(i - 1) % len(self.helpers)].get_model_weights() # Get weights of previous helper
                    helper.set_model_weights(weights)
                    model = helper.get_model()

                    converted_states, converted_actions = training_inputs[i]
                    minibatch_size: int = len(converted_states)//2 # Hard-coded batch size to be half of training set
                    
                    history = model.fit(converted_states, converted_actions, epochs=train_epochs, batch_size=minibatch_size, callbacks=[callback], verbose=verbose)

                    histories[i].append(history)

                self.info["training_iterations"] = iter
                # Custom Early Stopping
                self.update_helpers_weights()
                if self._check_planning_success():
                    consecutive_solved += 1
                    if consecutive_solved >= 20:
                        print(f"Reached goal in {20} consecutive iterations.")
                        self.info["early_solving"] = True
                        break
                else:
                    consecutive_solved = 0
                    self.save_intermediary_weights(iter, save_path)
        except KeyboardInterrupt:
            self.info["early_stopped"] = True
        
        return histories


    def train(self, exploration_loops: int = 550, train_epochs: int = 100, verbose: int = 0, policy_exploration: bool=True, save_path: str='/data/models') -> list:
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
        self.info['policy_exploration'] = policy_exploration

        histories: list = [[]] * len(self.helpers)
        self.info["early_solving"] = False
        self.info["early_stopped"] = False
        tic = time.process_time()

        if self.info['policy_exploration']:
            histories = self.train_with_exploration(exploration_loops, train_epochs, verbose, save_path)
        else:
            histories = self.train_without_exploration(exploration_loops, train_epochs, verbose, save_path)

        toc = time.process_time()
        self.info["training_time"] = toc-tic
        
        self.update_helpers_weights()
        shared_weights = self.helpers[0].get_model_weights()
        return histories, shared_weights


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def train(domain, problems, valid: str='', layers: int=2, policy_exploration: bool=False, save: str='', verbose: int=0):
    if save == '':
        save = domain.split('/')[-2]
    try: # Creates directory for saving intermediary models
        os.mkdir(f'data/{save}')
    except:
        pass
    
    print("Instancing ASNets")
    trainer = Trainer(domain, problems, valid, asnet_layers=layers)
    print("Starting Training...")
    _, weights = trainer.train(verbose=verbose, policy_exploration=policy_exploration, save_path=f'data/{save}')
    print(f"Training concluded in {trainer.info['training_time']}s")

    
    with open(f'data/{save}.json', 'w') as f:
        json.dump(weights, f, cls=NumpyEncoder)
    with open(f'info/{save}.json', 'w') as f:
        json.dump(trainer.info, f, cls=NumpyEncoder)


@click.command()
@click.option("--domain", "-d", type=str, help="Path to the problem's domain PPDDL file.", default='problems/blocksworld/domain.pddl')
@click.option(
    "--problems", "-p", type=str, help="Path to (multiple) problem's instance PPDDL files.", multiple=True,
    default=[
        'problems/blocksworld/pb3_p0.pddl',
        'problems/blocksworld/pb3_p1.pddl',
        'problems/blocksworld/pb3_p2.pddl',
        'problems/blocksworld/pb4_p0.pddl',
        'problems/blocksworld/pb5_p0.pddl',
        'problems/blocksworld/pb5_p1.pddl',
        'problems/blocksworld/pb5_p2.pddl'
    ])
@click.option("--valid", "-v", type=str, help="Path to problem's instance PPDDL files used for training's validation.", default='')
@click.option("--layers", "-l", type=int, help="Number of layers of the trained ASNets", default=2)
@click.option("--policy_exploration", "-pe", type=bool, help="If the training should include exploration guided by the trained ASNet's policy or not.", default=False)
@click.option("--save", "-s", type=str, help="Name of file with saved weights after training. If none is set, used the domain's name.", default='')
@click.option("--verbose", type=int, help="Debug prints. Off by default.", default=0)
def execute(domain, problems, valid: str, layers: int, policy_exploration: bool, save: str, verbose: int):
    train(domain, problems, valid, layers, policy_exploration, save, verbose)



if __name__ == "__main__":
    execute()
