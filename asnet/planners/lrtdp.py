"""Implementation of the LRTDP algorithm such as described in the paper
"Labeled RTDP: Improving the Convergence of Real-Time Dynamic Programming"

The following code used inspiration from markkho's msdm package implementation
(https://github.com/markkho/msdm/blob/master/msdm/algorithms/lrtdp.py)
"""
import random

from .rtdp import RTDP, is_goal
from ..heuristics.null_heuristic import NullHeuristic

from ippddl_parser.parser import Parser


DEBUG: bool = False



class LRTDP(RTDP):
    def __init__(self,
                 parser: Parser,
                 heuristic=NullHeuristic(),
                 discount_rate: float=0.9,
                 bellman_error_margin: float=1e-2,
                 iterations: int=int(2**30),
                 max_trial_length: int = None
                ) -> None:
        """Labeled Real-Time Dynamic Programming (Bonet & Geffner 2003)."""
        super().__init__(parser, heuristic, discount_rate, bellman_error_margin, iterations)
        self.solved_states: dict = {}

        if max_trial_length:
            self.MAX_TRIAL_LENGTH = max_trial_length
    

    def _initialize_values(self):
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
            # Sets all terminal/absorbing states as solved
            if len(self._get_applicable_actions(s)) == 0:
                self.solved_states[s] = True
    

    def execute(self):
        self._initialize_values()
        
        for i in range(self.iterations):
            if DEBUG:
                print("Iteration: ", i)
                for state, val in self.state_values.items():
                    if not self.solved_states[state]:
                        print(f"{state}: {val}")
            if all(self.solved_states[s] for s in self.states):
                return
            random_state = self.states[random.randint(0, len(self.states) - 1)]
            self.trial(random_state)
        if i == (self.iterations - 1):
            print(f"LRTDP not converged after {self.iterations} iterations")


    def trial(self, s):
        # Ghallab, Nau, Traverso: Algorithm 6.17
        visited = [s, ]
        while not self.solved_states[s]:
            act = self.policy(s)
            self._bellman_update(s)
            s = act.apply(s) # Applies action with best Q value to state

            visited.append(s)
            
            if len(visited) > self.MAX_TRIAL_LENGTH:
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
            residual = self.state_values[s] - self.Q(s, self.policy(s))
            if abs(residual) > self.bellman_error_margin:
                flag = False
            else:
                successor_states, _ = self._get_successor_states(s)
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



if __name__ == "__main__":
    domain_file = 'problems/blocksworld/domain.pddl'
    problem_file = 'problems/blocksworld/pb5_p0.pddl'

    parser: Parser = Parser()
    parser.scan_tokens(domain_file)
    parser.scan_tokens(problem_file)
    parser.parse_domain(domain_file)
    parser.parse_problem(problem_file)

    lrtdp = LRTDP(parser)
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