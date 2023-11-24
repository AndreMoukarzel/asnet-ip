"""Implementation of the LRTDP algorithm such as described in the paper
"Labeled RTDP: Improving the Convergence of Real-Time Dynamic Programming"

The following code used inspiration from markkho's msdm package implementation
(https://github.com/markkho/msdm/blob/master/msdm/algorithms/lrtdp.py)
"""
import random

from ..heuristics.null_heuristic import NullHeuristic

from ippddl_parser.parser import Parser
from ippddl_parser.value_iteration import ValueIterator


def is_goal(state, positive_goals: frozenset, negative_goals: frozenset) -> bool:
    """Returns if the specified state is a goal state"""
    if positive_goals.issubset(state) and negative_goals.isdisjoint(state):
        return True
    return False


def get_applicable_actions(state, actions, positive_goals, negative_goals):
    if is_goal(state, positive_goals, negative_goals):
        return []
    
    applicable = []
    for act in actions:
        if act.is_applicable(state):
            applicable.append(act)
    return applicable


def get_successor_states(state, actions):
    successor_states = []
    states_probs = []
    for act in actions:
        if act.is_applicable(state):
            future_states, probs = act.get_possible_resulting_states(state)
            for i, state in enumerate(future_states): 
                successor_states.append(state)
                states_probs.append(probs[i])
    return successor_states, states_probs


class LRTDP:
    def __init__(self,
                 parser: Parser,
                 heuristic=NullHeuristic(),
                 discount_rate: float=0.9,
                 bellman_error_margin: float=1e-2,
                 iterations: int=int(2**30)
                ) -> None:
        """
        Labeled Real-Time Dynamic Programming (Bonet & Geffner 2003).

        Parameters
        ----------
        heuristic : Callable[[HashableState], float]
            State-heuristic function. If this over-estimates
            the value at all states, then the
            algorithm will converge to an optimal solution.
        iterations : int
            Number of trials of LRTDP to run.
        """
        self.parser = parser
        self.heuristic = heuristic
        self.discount_rate = discount_rate
        self.bellman_error_margin = bellman_error_margin
        self.iterations = iterations
        self.actions: list = self._get_all_actions()
        self.states: list = list(self._get_all_states(self.actions))

        self.randomize_action_order = True
        self.max_trial_length = float('inf')
    

    def _get_all_actions(self):
        ground_actions = []
        for action in self.parser.actions:
            for act in action.groundify(self.parser.objects, self.parser.types):
                # Does not add actions invalidated by equality preconditions
                invalid_equalty: bool = False
                for precond in act.negative_preconditions:
                    if precond[0] == 'equal' and precond[1] == precond[2]:
                        invalid_equalty = True
                
                if not invalid_equalty:
                    ground_actions.append(act)
        return ground_actions
    

    def _get_all_states(self, all_actions) -> set:
        iterator = ValueIterator()
        return iterator.get_all_states(self.parser.state, all_actions)
    

    def _get_applicable_actions(self, state):
        if state in self.applicable_actions:
            if self.randomize_action_order:
                # Randomizes order so actions with same Q value may all be picked
                random.shuffle(self.applicable_actions[state])
            return self.applicable_actions[state]
        
        app_actions = get_applicable_actions(state, self.actions, self.parser.positive_goals, self.parser.negative_goals)
        self.applicable_actions[state] = app_actions
        if self.randomize_action_order:
            # Randomizes order so actions with same Q value may all be picked
            random.shuffle(self.applicable_actions[state])
        return self.applicable_actions[state]


    def execute(self):
        self.state_values: dict = {} # V in the Bellman equation
        self.solved_states: dict = {}
        self.applicable_actions: dict = {}

        for s in self.states:
            # Initiates states values as the heuristic value.
            h_val: float = self.heuristic(s)
            if not isinstance(h_val, float):
                # Some heuristics return additional information besides the heuristic value
                h_val = h_val[0]
            # We use negative values because heuristic represent costs
            # and we represent the states values by their estimated reward.
            self.state_values[s] = -h_val
            self.solved_states[s] = False
        
        for i in range(self.iterations):
            if all(self.solved_states[s] for s in self.states):
                return
            random_state = self.states[random.randint(0, len(self.states) - 1)]
            self.lrtdp_trial(random_state)
        if i == (self.iterations - 1):
            #warnings.warn(f"LRTDP not converged after {iterations} iterations")
            print(f"LRTDP not converged after {self.iterations} iterations")


    def lrtdp_trial(self, s):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        visited = [s, ]
        while not self.solved_states[s]:
            app_acts = self._get_applicable_actions(s)
            if len(app_acts) == 0:
                # If there are no successor states (aka the state is absorbing) we consider the state solved.
                self.solved_states[s] = True
                break

            self._bellman_update(s)
            act = self.policy(s)

            s = act.apply(s) # Applies action with best Q value to state
            visited.append(s)
            
            if len(visited) > len(self.states) * 10:#self.max_trial_length:
                break
        s = visited.pop()
        while self._check_solved(s) and visited:
            s = visited.pop()


    def _check_solved(self, s):
        # GNT Algorithm 6.18
        flag = True
        open = []
        closed = []
        if not self.solved_states[s]:
            open.append(s)
        while open:
            s = open.pop()
            closed.append(s)
            if len(self._get_applicable_actions(s)) == 0:
                flag = False
                continue
            residual = self.state_values[s] - self.Q(s, self.policy(s))
            if abs(residual) > self.bellman_error_margin:
                flag = False
            else:
                successor_states, _ = get_successor_states(s, self.actions)
                for ns in successor_states:
                    if not self.solved_states[ns] and ns not in open and ns not in closed:
                        open.append(ns)
        if flag:
            for ns in closed:
                self.solved_states[ns] = True
        else:
            while closed:
                s = closed.pop()
                self._bellman_update(s)
        return flag


    def _bellman_update(self, s):
        '''
        Following Bonet & Geffner 2003, we only explicitly store value
        and compute Q values and the policy from it, important in computing
        the residual in _check_solved().
        '''
        app_acts = self._get_applicable_actions(s)
        self.state_values[s] = max(self.Q(s, a) for a in app_acts)


    def Q(self, s, a):
        sucessors, _ = get_successor_states(s, self.actions)
        if sucessors is None:
            # If there are no successor states (aka the state is absorbing) we consider the state solved.
            return 0
        q = 0
        future_states, probs = a.get_possible_resulting_states(s)
        for i, ns in enumerate(future_states):
            reward: float = 0.0
            if is_goal(ns, self.parser.positive_goals, self.parser.negative_goals):
                reward = 10.0
            q += probs[i] * (reward + self.discount_rate * self.state_values[ns])
        return q


    def policy(self, s):
        action_list = get_applicable_actions(s, self.actions, self.parser.positive_goals, self.parser.negative_goals)
        if len(action_list) == 0:
            return None
        return max(action_list, key=lambda a: self.Q(s, a))
    

    def solution_is_valid(self, solution_range:int = 50) -> bool:
        s = self.parser.state
        for _ in range(solution_range):
            if is_goal(s, self.parser.positive_goals, self.parser.negative_goals):
                return True
            act = self.policy(s)
            s = act.apply(s)
        return False


if __name__ == "__main__":
    from ..heuristics.lm_cut import LMCutHeuristic
    from ..heuristics.hmax import HMax

    domain_file = 'problems/deterministic_blocksworld/domain.pddl'
    problem_file = 'problems/deterministic_blocksworld/pb3.pddl'

    parser: Parser = Parser()
    parser.scan_tokens(domain_file)
    parser.scan_tokens(problem_file)
    parser.parse_domain(domain_file)
    parser.parse_problem(problem_file)

    lmcut = LMCutHeuristic(parser)
    hmax = HMax(parser)

    lrtdp = LRTDP(parser, lmcut)
    lrtdp.execute()

    s = parser.state
    print(s)
    for _ in range(20):
        if is_goal(s, parser.positive_goals, parser.negative_goals):
            print("GOAL REACHED")
            break
        act = lrtdp.policy(s)
        s = act.apply(s)
        print(f'|-> {act.name}[{act.parameters}] -> {s}')
