from typing import List, Tuple

from asnet import ASNet

import numpy as np
from tensorflow import keras
from ippddl_parser.value_iteration import ValueIterator


class Trainer:

    def __init__(self, domain_file: str, problem_file: str) -> None:
        self.net = ASNet(domain_file, problem_file)
        self.parser = self.net.parser
        self.init_state: str = self.parser.state

        # Creates Action instances referenced by the names used in ground_actions.
        # Used when checking if an action is applicable in a state
        self.action_objects = {}
        self.instance_Actions: list = self.net._get_ground_actions() # List of Action objects representing all ground actions of instance
        for act in self.instance_Actions:
            self.action_objects[(act.name, act.parameters)] = act
        
        # Calculates value of each state of the problem
        iterator = ValueIterator()
        self.all_states = iterator.get_all_states(self.parser.state, self.instance_Actions)
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
        for state in self.all_states:
            best_action: tuple = ()
            best_state_val: float = -999.0
            for act in self.instance_Actions:
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
    

    def _action_to_index(self, action: tuple) -> int:
        """Given an action tuple of (action name, action parameters), returns
        its index in the currently instanced ASNet
        """
        return self.net.ground_actions.index(action)
    

    def _index_to_action(self, index: int) -> tuple:
        """Given an action index in the currently instanced ASNet, returns the
        corresponding action tuple of (action name, action parameters)
        """
        return self.net.ground_actions[index]


    def _action_to_output(self, action_index: int) -> List[float]:
        """Converts an action index to the equivalent output of a categorical
        neural network. That is, a list of values where the chosen action has
        value 1.0
        """
        action_num: int = len(self.net.ground_actions)
        action_output : List[float] = [0.0] * action_num
        action_output[action_index] = 1.0
        return action_output
    

    def is_goal(self, state: str) -> bool:
        """Returns if the current state is a goal state"""
        goal_pos: frozenset = self.parser.positive_goals
        goal_neg: frozenset = self.parser.negative_goals
        if goal_pos.issubset(state) and goal_neg.isdisjoint(state):
            return True
        return False
    

    def applicable_actions(self, state: str) -> list:
        """Returns a list of applicable actions for the state"""
        if self.is_goal(state):
            return []
        
        app_acts: list = [] # List of Action objects
        for act in self.instance_Actions:
            if act.is_applicable(state):
                app_acts.append(act)
        return app_acts


    def run_policy(self, initial_state: str, max_steps: int = 500) -> Tuple[List[str], List[tuple]]:
        """Executes the ASNet's chosen actions in the problem instance until
        a terminal state is found or the number of maximum allowed steps is
        reached.

        Returns the list of encountered states and the list of taken actions
        """
        states: List[str] = [initial_state]
        actions: List[tuple] = []
        model = self.net.model
        curr_state: str = initial_state
        app_actions: list = self.applicable_actions(curr_state)
        
        while not self.is_goal(curr_state) and len(app_actions) > 0:
            input_state: List[float] = self._state_to_input(curr_state)
            action_probs: np.ndarray = model.predict((input_state,))[0]
            max_prob: float = action_probs.max()
            chosen_action_index = np.where(action_probs == max_prob)[0][0]
            chosen_action: tuple = self._index_to_action(chosen_action_index)
            actions.append(chosen_action)

            # Applies chosen action to state, if applicable
            action_applied: bool = False
            for act in app_actions:
                if (act.name, act.parameters) == chosen_action:
                    action_applied = True
                    curr_state = act.apply(curr_state)
                    states.append(curr_state)
                    app_actions = self.applicable_actions(curr_state)
            if not action_applied:
                return states, actions

        return states, actions


    def teacher_rollout(self, initial_state, all_states: List[str], state_best_actions: List[tuple]) -> List[str]:
        """Rollouts the teacher's policy from the given initial state until
        a goal or terminal state is reached

        Returns the list of encountered states
        """
        states: List[str] = [initial_state]
        curr_state: str = initial_state
        app_actions: list = self.applicable_actions(curr_state)
        
        while not self.is_goal(curr_state) and len(app_actions) > 0:
            state_index: int = all_states.index(curr_state)
            chosen_action: tuple = state_best_actions[state_index]

            # Applies chosen action to state, if applicable
            action_applied: bool = False
            for act in app_actions:
                if (act.name, act.parameters) == chosen_action:
                    action_applied = True
                    curr_state = act.apply(curr_state)
                    states.append(curr_state)
                    app_actions = self.applicable_actions(curr_state)
            if not action_applied:
                return states
        
        return states


    def train(self, full_epochs: int = 50, train_epochs: int = 100):
        """Trains the instanced ASNet on the problem"""
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

        # Gets inputs and outputs to train the model
        states: List[str] = list(self.all_states)
        state_best_actions: List[tuple] = self._best_actions_by_state()

        for _ in range(full_epochs):
            # Runs policy in search of states to be explored in training
            explored_states: List[str] = self.run_policy(self.init_state)[0]
            rollout_states: List[str] = []
            # Rollouts the teacher policy from each state found from the model's run,
            # to be certain that there will be optimal states in the training
            for state in explored_states:
                rollout_states += self.teacher_rollout(state, states, state_best_actions)
            training_states: List[str] = explored_states + rollout_states

            # Gets "correct" action according to Teacher Planner for each selected state
            # and converts states and actions to format inputable in model
            correct_actions: List[tuple] = [state_best_actions[states.index(s)] for s in training_states]
            converted_states: List[List[float]] = [self._state_to_input(s) for s in training_states]
            correct_actions_indexes: List[int] = [self._action_to_index(act) for act in correct_actions]
            converted_actions: List[List[float]] = [self._action_to_output(act_ind) for act_ind in correct_actions_indexes]

            minibatch_size: int = len(training_states)//2 # Hard-coded batch size to be half of training set
            model.fit(converted_states, converted_actions, epochs=train_epochs, batch_size=minibatch_size)



if __name__ == "__main__":
    domain = '../problems/deterministic_blocksworld/domain.pddl'
    problem = '../problems/deterministic_blocksworld/pb5.pddl'

    trainer = Trainer(domain, problem)
    trainer.train(full_epochs=15, train_epochs=200)
    print("Executing a test policy with the Network")
    states, actions = trainer.run_policy(trainer.init_state)
    for i, act in enumerate(actions):
        print("State: ", states[i])
        print("Action taken: ", act)
    print("State: ", states[-1])
