from typing import List

from asnet import ASNet

from tensorflow import keras
from ippddl_parser.value_iteration import ValueIterator


class Trainer:

    def __init__(self, domain_file: str, problem_file: str) -> None:
        self.net = ASNet(domain_file, problem_file)
        self.parser = self.net.parser
        self.init_state = self.parser.state

        # Creates Action instances referenced by the names used in ground_actions.
        # Used when checking if an action is applicable in a state
        self.action_objects = {}
        ground_acts = self.net._get_ground_actions()
        for act in ground_acts:
            self.action_objects[(act.name, act.parameters)] = act
        
        # Calculates value of each state of the problem
        iterator = ValueIterator()
        self.all_states = iterator.get_all_states(self.parser.state, ground_acts)
        self.state_vals = iterator.solve(domain_file, problem_file)
    

    def _state_to_input(self, state) -> List[float]:
        """Converts a state to a list of 0s and 1s that can be received as an
        input by the instanced ASNet
        """
        state_input: List[float] = []

        for act in self.net.ground_actions:
            # For each related proposition of an action, adds a value
            # indicating if it is in the current state
            for prop_index in self.net.pred_indexed_relations[act]:
                current_prop = self.net.propositions[prop_index]
                if current_prop in state:
                    state_input.append(1.0)
                else:
                    state_input.append(0.0)
            # For each related proposition of an action, adds a value
            # indicating if it is in a goal state
            for prop_index in self.net.pred_indexed_relations[act]:
                current_prop = self.net.propositions[prop_index]
                if current_prop in self.parser.positive_goals:
                    state_input.append(1.0)
                else:
                    state_input.append(0.0)
            # Adds final value indicating if the action is applicable
            if self.action_objects[act].is_applicable(state):
                state_input.append(1.0)
            else:
                state_input.append(0.0)

        return state_input


    def _best_actions_by_state(self) -> List[tuple]:
        """Given the calculated value of each state, gets the expected output
        action in each given state in self.all_states.
        
        Returns a list of tuples of format (action name, action parameters)
        """
        state_values: List[float] = []
        for state in self.all_states:
            state_val: float = self.state_vals[state]
            state_values.append(state_val)
        
        state_best_actions: List[tuple] = []
        ground_actions = self.net._get_ground_actions()
        for state in self.all_states:
            best_action: tuple = ()
            best_state_val: float = -999.0
            for act in ground_actions:
                if act.is_applicable(state):
                    # "Future" states are the s', the states reached by applying the action to current state s 
                    future_states, _ = act.get_possible_resulting_states(state)

                    # Gets the most valuable result among future states
                    best_fut_val: float = -999.0
                    for fut_state in future_states:
                        fut_state_index: int = list(self.all_states).index(fut_state)
                        fut_state_val: float = state_values[fut_state_index]
                        best_fut_val = max(best_fut_val, fut_state_val)
                            
                    if best_action == () or best_fut_val >= best_state_val:
                        best_state_val = best_fut_val
                        best_action = (act.name, act.parameters)
            
            state_best_actions.append(best_action)
        
        return state_best_actions


    def _action_to_output(self, action_index: int) -> List[float]:
        """Converts an action index to the equivalent output of a categorical
        neural network. That is, a list of values where the chosen action has
        value 1.0
        """
        action_num: int = len(self.net.ground_actions)
        action_output : List[float] = [0.0] * action_num
        action_output[action_index] = 1.0
        return action_output
    

    def train(self):
        """Trains the instanced ASNet on the problem"""
        # Gets inputs and outputs to train the model
        states: List[str] = list(self.all_states)
        state_best_actions: List[tuple] = self._best_actions_by_state()

        # Converts states and actions to format inputable in model
        converted_states: List[List[float]] = [self._state_to_input(state) for state in states]
        best_actions_indexes: List[int] = [self.net.ground_actions.index(act) for act in state_best_actions]
        converted_best_actions: List[List[float]] = [self._action_to_output(act_ind) for act_ind in best_actions_indexes]

        model = self.net.model
        model.compile(
            loss=keras.losses.categorical_crossentropy, # Loss function is logloss
            optimizer=keras.optimizers.SGD(learning_rate=0.0005),
            metrics=[
                'accuracy',
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                'MeanSquaredError',
                'AUC'
            ]
        )

        model.fit(converted_states, converted_best_actions)



if __name__ == "__main__":
    domain = '../problems/deterministic_blocksworld/domain.pddl'
    problem = '../problems/deterministic_blocksworld/pb1.pddl'

    trainer = Trainer(domain, problem)
    trainer.train()
