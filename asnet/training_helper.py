from typing import List, Tuple, Dict, FrozenSet
import time

from .asnet import ASNet
from .asnet_no_lmcut import ASNetNoLMCut
from .heuristics.lm_cut import LMCutHeuristic
from .planners.lrtdp import LRTDP
from .planners.stlrtdp import STLRTDP
from .weight_transfer import get_lifted_weights, set_lifted_weights

import numpy as np
from ippddl_parser.parser import Parser
from ippddl_parser.value_iteration import ValueIterator


class TrainingHelper:
    """
    Class with helpful methods for training. Instances both the ASNet as its
    teacher planner into itself.

    Methods
    -------
    is_goal(state)
        Returns if a state is a goal state

    applicable_actions(state)
        Returns a list of all applicable actions in the state

    run_policy(initial_state, max_steps, verbose)
        Runs the ASNet's policy from the initial_state until a goal state or max_steps are reached.

    teacher_rollout(initial_state, all_states, state_best_actions)
        Rollouts the teacher planner from the initial_state until a terminal state is reached.

    generate_training_inputs(verbose)
        Generates inputs compatible with ASNet's training.
    
    get_model()
        Returns the ASNet's model, if it was instanced.

    get_model_weights() -> Dict[str, np.array]
        Returns the instanced ASNet's weights, identifying them by the lifted name of their related action/proposition.

    set_model_weights(weights: Dict[str, np.array]) -> None
        Overwrite the instanced ASNet's weights with the specified weights.
    """

    def __init__(self, domain_file: str, problem_file: str, instance_asnet: bool = True, lmcut: bool = True, solve: bool=True) -> None:
        """
        Parameters
        ----------
        domain_file : str
            IPPDDL file specifying the problem domain
        problem_file : str
            IPPDDL file specifying the problem instance
        instance_asnet: bool, Optional
            If it should instance an ASNet's network. By default is True.
        lmcut: bool, Optional
            If the instanced ASNet should use the LM-Cut heuristic. By default is true.
        """
        self.info: dict = {} # Saves information about training
        self.lmcut: bool = lmcut
        if lmcut:
            self.net: ASNet = ASNet(domain_file, problem_file, instance_network=instance_asnet)
            self.lm_heuristic = LMCutHeuristic(self.net.parser)
        else:
            self.net: ASNetNoLMCut = ASNetNoLMCut(domain_file, problem_file, instance_network=instance_asnet)
        if instance_asnet:
            self.net.compile()
        self.parser: Parser = self.net.parser
        self.init_state: Tuple[str] = self.parser.state

        # Creates Action instances referenced by the names used in ground_actions.
        # Used when checking if an action is applicable in a state
        self.action_objects: Dict[Tuple[str], any] = {}
        self.instance_Actions: list = self.net._get_ground_actions() # List of Action objects representing all ground actions of instance
        for act in self.instance_Actions:
            self.action_objects[(act.name, act.parameters)] = act
        
        if solve:
            # Calculates value of each state of the problem with the appropriate solver
            tic = time.process_time()
            if not lmcut:
                self.solver = ValueIterator()
                self.info['solver'] = 'VI'
                self.all_states = self.solver.get_all_states(self.parser.state, self.instance_Actions)
                self.solver.solve(domain_file, problem_file)
            else:
                self.solver = LRTDP(self.parser)
                self.info['solver'] = 'LRTDP'
                if ':imprecise' in self.parser.requirements:
                    self.solver = STLRTDP(self.parser)
                    self.info['solver'] = 'STLRTDP'
                print(f"Solving problem instance [{problem_file.split('/')[-1]}]")
                self.solver.execute()
                for _ in range(5):
                    if self.solver.solution_is_valid():
                        break
                    self.solver.execute()
                if not self.solver.solution_is_valid():
                    raise RuntimeError("Teacher planner could not find solution to problem in the allotted trials")
                print("Problem instance solved!")
                
                self.all_states = self.solver.states
            toc = time.process_time()
            self.info['time_to_solve'] = toc-tic
    

    ################################################# PRIVATE METHODS ##################################################


    def _state_to_input(self, state: FrozenSet[Tuple[str]]) -> List[float]:
        """Converts a state to a list of 0s and 1s that can be received as an
        input by the instanced ASNet or ASNetNoLMCut.

        Parameters
        ----------
        state: FrozenSet[Tuple[str]]
            A set of tuples of format (predicate name, object1, object2...)
            with all true propositions representing the state.
        
        Returns
        -------
        List[float]
            Inputable representation of the received state.
        """
        state_input: List[float] = []

        for act in self.net.ground_actions:
            # For each related proposition of an action, adds a value
            # indicating if it is in the current state
            for prop_index in self.net.get_related_prop_indexes()[act]:
                current_prop = self.net.propositions[prop_index]
                if current_prop in state:
                    state_input.append(1.0)
                else:
                    state_input.append(0.0)
            # For each related proposition of an action, adds a value
            # indicating if it is in a goal state
            for prop_index in self.net.get_related_prop_indexes()[act]:
                current_prop = self.net.propositions[prop_index]
                if current_prop in self.parser.positive_goals:
                    state_input.append(1.0)
                else:
                    state_input.append(0.0)
            # Adds values related to the LM-Cut heuristic
            if self.lmcut:
                cuts = self.lm_heuristic.all_cuts
                state_vals: List[float] = [0.0, 0.0, 1.0]
                for cut_group in cuts:
                    for cut_act in cut_group:
                        if cut_act == act:
                            state_vals[2] = 0.0

                            if len(cut_group) == 1:
                                state_vals[0] = 1.0
                            else:
                                state_vals[1] = 1.0
                state_input.append(state_vals[0]) # Iff a appears as the only action in at least one landmark
                state_input.append(state_vals[1]) # Iff a appears in a landmark containing two or more actions
                state_input.append(state_vals[2]) # Iff does not appear in any landmark
            # Adds final value indicating if the action is applicable
            if self.action_objects[act].is_applicable(state):
                state_input.append(1.0)
            else:
                state_input.append(0.0)

        return state_input


    def _best_actions_by_state(self) -> List[Tuple[str]]:
        """For all possible states, returns their respective 'optimal action' as
        defined by the teacher planner.
        
        Returns
        -------
        List[Tuple[str]]
            Tuples of format (action name, action parameters) representing the
            best actions for each state in the order they appear in
            self.all_states.
        """
        state_best_actions: List[tuple] = [self.solver.policy(state) for state in self.all_states]
        default_act = self.instance_Actions[0] # Action defaulted to when no actions are applicable in a state
        return [(act.name, act.parameters) if act is not None else (default_act.name, default_act.parameters) for act in state_best_actions]
    

    def _action_to_index(self, action: Tuple[str]) -> int:
        """Given an action tuple of (action name, action parameters), returns
        its index in the currently instanced ASNet
        """
        return self.net.get_ground_actions().index(action)
    

    def _index_to_action(self, index: int) -> Tuple[str]:
        """Given an action index in the currently instanced ASNet, returns the
        corresponding action tuple of (action name, action parameters)
        """
        return self.net.get_ground_actions()[index]


    def _action_to_output(self, action_index: int) -> List[float]:
        """Converts an action index to the equivalent output of a categorical
        neural network.
        
        Returns
        -------
        List[float]
            A list where the chosen action's index has value 1.0 and all other
            elements are 0.0
        """
        action_num: int = len(self.net.get_ground_actions())
        action_output : List[float] = [0.0] * action_num
        action_output[action_index] = 1.0
        return action_output
    

    ################################################## PUBLIC METHODS ##################################################
    

    def is_goal(self, state: FrozenSet[Tuple[str]]) -> bool:
        """Returns if the specified state is a goal state"""
        goal_pos: frozenset = self.parser.positive_goals
        goal_neg: frozenset = self.parser.negative_goals
        if goal_pos.issubset(state) and goal_neg.isdisjoint(state):
            return True
        return False
    

    def applicable_actions(self, state: FrozenSet[Tuple[str]]) -> list:
        """Returns a list of applicable actions for the state"""
        if self.is_goal(state):
            return []
        
        app_acts: list = [] # List of Action objects
        for act in self.instance_Actions:
            if act.is_applicable(state):
                app_acts.append(act)
        return app_acts


    def run_policy(
            self, initial_state: FrozenSet[Tuple[str]], model = None, max_steps: int = 50, verbose: int = 0
        ) -> Tuple[List[FrozenSet[Tuple[str]]], List[Tuple[str]]]:
        """Executes the ASNet's chosen actions in the problem instance until
        a terminal state is found or the number of maximum allowed steps is
        reached.


        Parameters
        ----------
        initial_state: FrozenSet[Tuple[str]]
            A set of tuples of format (predicate name, object1, object2...)
            with all true propositions representing the state.
        model : keras.Model
            Model to be used to run policy on problem instance. If none is given,
            uses the instanced model.
        max_steps: int, optional
            Maximum number of actions the ASNet will take before terminating the
            policy.
        verbose: int, optional
            How much information the ASNet will print when processing each chosen
            action. Can be 0, 1 or 2. By default is 0 (no prints).

        Returns
        -------
        Tuple[List[FrozenSet[Tuple[str]]], List[Tuple[str]]]
            Returns a list of the encountered states while executing the policy
            and another list with the actions taken in order.
        """
        states: List[str] = [initial_state]
        actions: List[tuple] = []
        curr_state: str = initial_state
        app_actions: list = self.applicable_actions(curr_state)

        if model is None:
            model = self.get_model()
        
        while not self.is_goal(curr_state) and len(app_actions) > 0:
            input_state: List[float] = self._state_to_input(curr_state)
            action_probs: np.ndarray = model.predict((input_state,), verbose=verbose)[0]
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
            if not action_applied or len(actions) >= max_steps:
                return states, actions

        return states, actions


    def teacher_rollout(
            self, initial_state: FrozenSet[Tuple[str]], all_states: List[FrozenSet[Tuple[str]]], state_best_actions: List[Tuple[str]]
        ) -> List[FrozenSet[Tuple[str]]]:
        """Rollouts the teacher planner's policy from the given initial state
        until a goal or terminal state is reached.

        Parameters
        ----------
        initial_state: FrozenSet[Tuple[str]]
            A set of tuples of format (predicate name, object1, object2...)
            with all true propositions representing the state.
        all_states: List[FrozenSet[Tuple[str]]]
            List with all possible states in the problem instance
        state_best_actions: List[Tuple[str]]
            List with the best action to take on each state. The value in index
            i is the best action to be taken when in state all_states[i].
        
        Returns
        -------
        List[FrozenSet[Tuple[str]]]
            All states in teacher planner's plan
        """
        states: List[FrozenSet[Tuple[str]]] = [initial_state]
        curr_state: FrozenSet[Tuple[str]] = initial_state
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
    

    def generate_training_inputs(self, model = None, verbose: int = 0) -> Tuple[List[List[float]], List[List[float]]]:
        """Generates the training states and their corresponding 'ideal' actions
        and converts them to a format inputable into an ASNet, so they can be
        used for training.

        Parameters
        ----------
        model : keras.Model, optional
            Model to be used to run policy on problem instance. If none is given,
            uses the instanced model.
        verbose: int, optional
            Verbosity mode when running the ASNet's policy. 0 = silent,
            1 = progress bar, 2 = one line per epoch.

        Returns
        -------
        Tuple[List[List[float]], List[List[float]]]
            The first returned list contains elements representing a state
            inputable into an ASNet

            The second returned list contains elements representing the desired
            output actions in the format outputed by an ASNet
        """
        states: List[str] = list(self.all_states)
        state_best_actions: List[tuple] = self._best_actions_by_state()
        # Runs policy in search of states to be explored in training
        explored_states: List[str] = self.run_policy(self.init_state, model, verbose)[0]
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

        return converted_states, converted_actions


    def get_model(self):
        """Returns the problem's ASNet model. If the network was not instanced,
        returns an error."""
        try:
            return self.net.get_model()
        except AttributeError:
            raise Exception("TrainingHelper's ASNet was not instanced!")


    def get_model_weights(self) -> Dict[str, np.array]:
        """Returns a dictionary with the lifted names of layers from the
        instanced ASNet as keys, and their weights and biases as values."""
        return get_lifted_weights(self.net)
    

    def set_model_weights(self, weights: Dict[str, np.array]) -> None:
        """Overwrites the weights and biases of the instanced ASNet by the
        received weights

        Parameters
        ----------
        weights: Dict[str, np.array]
            Weights and Biases such as obtained by the method get_model_weights().
            It is expected that the weights variable will be originary from
            a compatible ASNet, based on a problem of the same domain.
        """
        set_lifted_weights(self.net, weights)
