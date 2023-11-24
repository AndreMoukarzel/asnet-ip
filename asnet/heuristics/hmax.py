"""Implementation of the Hmax delete-relaxed heuristic."""
from heapq import heapify, heappush, heappop

from .delete_relaxation import RelaxedFact, JustificationGraph

from ippddl_parser.parser import Parser


class HMax(JustificationGraph):

    def _compute_hmax(self, state):
        """Navigates the justification graph to compute HMax values"""
        reachable: set = set()
        facts_seen: set = set()
        facts_cleared: set = set()
        operators_reset: set = set()
        starting_facts: set = {fact for fact in state}
        to_be_expanded = []
        heapify(to_be_expanded)

        if 'ALWAYS TRUE' in self.facts:
            starting_facts.add('ALWAYS TRUE')
        for fact in starting_facts:
            if fact in self.facts:
                Fact: RelaxedFact = self.facts[fact]
                Fact.heuristic_value = 0.0
                facts_seen.add(Fact)
                facts_cleared.add(Fact)
                heappush(to_be_expanded, Fact)

        while to_be_expanded:
            Fact: RelaxedFact = heappop(to_be_expanded)
            if Fact == self.facts["GOAL"]:
                self.dead_end = False
            
            reachable.add(Fact)
            hmax = Fact.heuristic_value
            # Updates all operators that have this RelaxedFact as precondition
            for operator in Fact.precondition_of:
                if operator not in operators_reset:
                    # If the operator was not seen in this computation iteration, it needs to have its value reset
                    operator.reset_cost()
                    operators_reset.add(operator)

                all_preconds_seen: bool = True
                for precond in operator.preconditions:
                    if precond not in facts_seen:
                        all_preconds_seen = False

                if all_preconds_seen:
                    # If all preconditions were already explored, update the operator's hmax value
                    if (operator.hval_origin is None) or (hmax > operator.hval_origin.heuristic_value):
                        operator.hval_origin = Fact
                        operator.heuristic_value = hmax + operator.cost
                    
                    hmax_new = operator.hval_origin.heuristic_value + operator.cost
                    for effect in operator.effects:
                        if effect not in facts_cleared:
                            effect.clear()
                            facts_cleared.add(effect)
                        effect.heuristic_value = min(hmax_new, effect.heuristic_value)
                        if effect not in facts_seen:
                            facts_seen.add(effect)
                            heappush(to_be_expanded, effect)
        self.reachable_facts = reachable
    
    
    def __call__(self, state):
        goal_state = self.facts["GOAL"]
        self._compute_hmax(state)
        return goal_state.heuristic_value


if __name__ == "__main__":
    domain_file = 'problems/deterministic_blocksworld/domain.pddl'
    problem_file = 'problems/deterministic_blocksworld/pb3.pddl'

    parser: Parser = Parser()
    parser.scan_tokens(domain_file)
    parser.scan_tokens(problem_file)
    parser.parse_domain(domain_file)
    parser.parse_problem(problem_file)

    hmax_heuristic = HMax(parser)
    print("Heuristic Value of initial state: ", hmax_heuristic(parser.state))
