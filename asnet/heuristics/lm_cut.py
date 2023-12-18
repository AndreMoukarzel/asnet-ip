"""Implementation of the Landmark Cut Heuristic such as described in the paper
"Landmarks, Critical Paths and Abstractions: What’s the Difference Anyway?"

The following code used inspiration from Pyperplan's implementation
(https://github.com/aibasel/pyperplan/blob/main/pyperplan/heuristics/lm_cut.py)
"""
from heapq import heappush, heappop

from .delete_relaxation import RelaxedFact
from .hmax import HMax

from ippddl_parser.parser import Parser


class LMCutHeuristic(HMax):

    def __init__(self, parser: Parser):
        super().__init__(parser)
        self.dead_end: bool = True
        self.goal_plateau: set = set()
        self.heuristic_value, self.all_cuts = self(parser.state)
    

    def _compute_hmax_from_last_cut(self, last_cut):
        """This computes hmax values starting from the last cut.

        This saves us from recomputing the hmax values of all facts/operators
        that have not changed anyway.
        NOTE: a complete cut procedure needs to be finished (i.e. one cut must
        be computed) for this to work!
        """
        to_be_expanded = []
        # add all operators from the last cut
        # to the queue of operators for which the hmax value needs to be
        # re-evaluated
        for op in last_cut:
            op.heuristic_value = op.hval_origin.heuristic_value + op.cost
            heappush(to_be_expanded, op)
        while to_be_expanded:
            # iterate over all operators whose effects might need updating
            op = heappop(to_be_expanded)
            next_hmax = op.heuristic_value
            # op_seen.add(op)
            for fact_obj in op.effects:
                # if hmax value of this fact is outdated
                fact_hmax = fact_obj.heuristic_value
                if fact_hmax > next_hmax:
                    # update hmax value
                    fact_obj.heuristic_value = next_hmax
                    # enqueue all ops of which fact_obj is a hmax origin
                    for next_op in fact_obj.precondition_of:
                        if next_op.hval_origin == fact_obj:
                            next_op.heuristic_value = next_hmax + next_op.cost
                            for supp in next_op.preconditions:
                                if supp.heuristic_value + next_op.cost > next_op.heuristic_value:
                                    next_op.hval_origin = supp
                                    next_op.heuristic_value = supp.heuristic_value + next_op.cost
                            heappush(to_be_expanded, next_op)
    

    def _compute_goal_plateau(self, fact_name):
        """Recursively mark a goal plateau."""
        # assure the fact itself is not in an unreachable region
        Fact = self.facts[fact_name]
        if Fact in self.reachable_facts and Fact not in self.goal_plateau:
            # add this fact to the goal plateau
            self.goal_plateau.add(Fact)
            for op in Fact.effect_of:
                # recursive call to mark hmax_supporters of all operators
                if op.cost == 0:
                    self._compute_goal_plateau(op.hval_origin.name)


    def find_cut(self, state):
        """This returns the set of relaxed operators which are in the landmark cut."""
        to_be_expanded = []
        facts_seen = set()
        op_cleared = set()
        cut = set()
        starting_facts: set = {x for x in state}

        if 'ALWAYS TRUE' in self.facts:
            starting_facts.add('ALWAYS TRUE')
        for fact in starting_facts:
            if fact in self.facts:
                Fact: RelaxedFact = self.facts[fact]
                facts_seen.add(Fact)
                Fact = self.facts[fact]
                heappush(to_be_expanded, Fact)
        
        while to_be_expanded:
            Fact = heappop(to_be_expanded)
            for operator in Fact.precondition_of:
                if not operator in op_cleared:
                    op_cleared.add(operator)
                
                all_preconds_seen: bool = True
                for precond in operator.preconditions:
                    if precond not in facts_seen:
                        all_preconds_seen = False

                if all_preconds_seen:
                    # If all preconditions were already explored, we can expand this operator
                    for eff in operator.effects:
                        if eff in facts_seen:
                            continue
                        if eff in self.goal_plateau:
                            cut.add(operator)
                        else:
                            facts_seen.add(eff)
                            heappush(to_be_expanded, eff)
        return cut
    

    def __call__(self, state):
        all_cuts = []
        heuristic_value = 0.0
        goal_state = self.facts["GOAL"]
        # reset dead end flag
        # --> asume node to be a dead end unless proven otherwise by the hmax
        # computation
        self.dead_end = True
        # next find all cuts
        # first compute hmax starting from the current state
        self._compute_hmax(state)
        if goal_state.heuristic_value == float("inf"):
            return float("inf"), []
        while goal_state.heuristic_value != 0:
            # next find an appropriate cut
            # first calculate the goal plateau
            self.goal_plateau.clear()
            self._compute_goal_plateau("GOAL")
            # then find the cut itself
            cut = self.find_cut(state)
            all_cuts.append(cut)
            # finally update heuristic value
            min_cost = min([o.cost for o in cut]) # The cost of the landmark
            heuristic_value += min_cost
            for o in cut:
                o.cost -= min_cost
            # compute next hmax
            self._compute_hmax_from_last_cut(cut)
        all_cuts_names = []
        for cut in all_cuts:
            all_cuts_names.append([operator.name for operator in cut])
        if self.dead_end:
            return float("inf"), all_cuts_names
        else:
            return heuristic_value, all_cuts_names


if __name__ == "__main__":
    domain_file = 'problems/deterministic_blocksworld/domain.pddl'
    problem_file = 'problems/deterministic_blocksworld/pb3_p0.pddl'

    parser: Parser = Parser()
    parser.scan_tokens(domain_file)
    parser.scan_tokens(problem_file)
    parser.parse_domain(domain_file)
    parser.parse_problem(problem_file)

    lmcut = LMCutHeuristic(parser)
    print("Heuristic Value of initial state: ", lmcut.heuristic_value)
    print("Landmark Cuts: ", lmcut.all_cuts)
