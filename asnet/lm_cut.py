"""Implementation of the Landmark Cut Heuristic such as described in the paper
"Landmarks, Critical Paths and Abstractions: What’s the Difference Anyway?"

The following code used inspiration from Pyperplan's implementation
(https://github.com/aibasel/pyperplan/blob/main/pyperplan/heuristics/lm_cut.py)
"""
from typing import List, Tuple, Dict
from heapq import heapify, heappush, heappop

from ippddl_parser.parser import Parser


class RelaxedFact:
    """Object representing a relaxed fact and its related operators in a
    justification graph
    """
    def __init__(self, name: Tuple[str]):
        self.name: Tuple[str] = name
        self.hmax_value: float = float("inf")
        self.precondition_of: List[RelaxedOperator] = []
        self.effect_of: List[RelaxedOperator] = []

    # Set comparison functions for a RelaxedFact
    def __lt__(self, other: "RelaxedFact"):
        return self.hmax_value < other.hmax_value

    def __leq__(self, other: "RelaxedFact"):
        return self.hmax_value <= other.hmax_value

    def __gt__(self, other: "RelaxedFact"):
        return self.hmax_value > other.hmax_value

    def __geq__(self, other: "RelaxedFact"):
        return self.hmax_value >= other.hmax_value
    
    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"Fact {self.name}, hmax: {self.hmax_value}" + \
            f"\n \t precond of: {[str(p) for p in self.precondition_of]}" + \
            f"\n \t effect of: {[str(p) for p in self.effect_of]}"

    def clear(self):
        self.hmax_value = float("inf")


class RelaxedOperator:
    def __init__(self, name: str, cost_zero: bool=False):
        self.name: str = name
        self.preconditions: List[RelaxedFact] = []
        self.effects: List[RelaxedFact] = []
        self.hmax_value: float = float("inf") # HMax value of the most expensive precondition
        self.hmax_origin: RelaxedFact = None # RelaxedFact from which the HMax was obtained
        self.cost: float = 1.0
        self.cost_zero: bool = cost_zero     
        if self.cost_zero:
            self.cost = 0.0
    

    def __lt__(self, other: "RelaxedOperator"):
        return self.hmax_value < other.hmax_value


    def reset_cost(self):
        self.cost = 1.0
        if self.cost_zero:
            self.cost = 0.0


class LMCutHeuristic:
    def __init__(self, parser: Parser):
        self.dead_end: bool = True
        self.goal_plateau: set = set()
        self.reachable_facts: set = set()
        self.facts: Dict[Tuple[str], RelaxedFact] = {}
        self.operators: Dict[str, RelaxedOperator] = {}
        # Removes all delete effects from considered actions, since this is a delete-relaxed heuristic algorithm.
        for action in parser.actions:
            for act in action.groundify(parser.objects, parser.types):
                act_name: Tuple[str] = (act.name, act.parameters)
                self.operators[act_name] = RelaxedOperator(act_name)

                if not act.positive_preconditions:
                    # If there are no preconditions, add generic fact that is always applicable
                    if not "ALWAYS TRUE" in self.facts:
                        self.facts["ALWAYS TRUE"] = RelaxedFact("ALWAYS TRUE")
                    self.add_precondition_to_operator(self.facts["ALWAYS TRUE"], self.operators[act_name])

                for precond in act.positive_preconditions:
                    if precond not in self.facts:
                        self.facts[precond] = RelaxedFact(precond)
                    self.add_precondition_to_operator(self.facts[precond], self.operators[act_name])
                
                for prob_effects in act.add_effects:
                    for effect in prob_effects:
                        if effect not in self.facts:
                            self.facts[effect] = RelaxedFact(effect)
                        self.add_effect_to_operator(self.facts[effect], self.operators[act_name])
        
        # Adds artificial unified goal and goal operator
        self.facts["GOAL"]: RelaxedFact = RelaxedFact("GOAL")
        self.operators["GOAL OPERATOR"]: RelaxedOperator = RelaxedOperator("GOAL OPERATOR", True)
        self.add_effect_to_operator(self.facts["GOAL"], self.operators["GOAL OPERATOR"])
        for fact in parser.positive_goals:
            assert fact in self.facts
            self.add_precondition_to_operator(self.facts[fact], self.operators["GOAL OPERATOR"])
        
        self.heuristic_value, self.all_cuts = self(parser.state)
    

    @staticmethod
    def add_precondition_to_operator(precondition: RelaxedFact, operator: RelaxedOperator):
        if precondition not in operator.preconditions:
            operator.preconditions.append(precondition)
        if operator not in precondition.precondition_of:
            precondition.precondition_of.append(operator)


    @staticmethod
    def add_effect_to_operator(effect: RelaxedFact, operator: RelaxedOperator):
        if effect not in operator.effects:
            operator.effects.append(effect)
        if operator not in effect.effect_of:
            effect.effect_of.append(operator)
    

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
                Fact.hmax_value = 0.0
                facts_seen.add(Fact)
                facts_cleared.add(Fact)
                heappush(to_be_expanded, Fact)

        while to_be_expanded:
            Fact: RelaxedFact = heappop(to_be_expanded)
            if Fact == self.facts["GOAL"]:
                self.dead_end = False
            
            reachable.add(Fact)
            hmax = Fact.hmax_value
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
                    if (operator.hmax_origin is None) or (hmax > operator.hmax_origin.hmax_value):
                        operator.hmax_origin = Fact
                        operator.hmax_value = hmax + operator.cost
                    
                    hmax_new = operator.hmax_origin.hmax_value + operator.cost
                    for effect in operator.effects:
                        if effect not in facts_cleared:
                            effect.clear()
                            facts_cleared.add(effect)
                        effect.hmax_value = min(hmax_new, effect.hmax_value)
                        if effect not in facts_seen:
                            facts_seen.add(effect)
                            heappush(to_be_expanded, effect)

        self.reachable_facts = reachable
    

    def _compute_hmax_from_last_cut(self, state, last_cut):
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
            op.hmax_value = op.hmax_origin.hmax_value + op.cost
            heappush(to_be_expanded, op)
        while to_be_expanded:
            # iterate over all operators whose effects might need updating
            op = heappop(to_be_expanded)
            next_hmax = op.hmax_value
            # op_seen.add(op)
            for fact_obj in op.effects:
                # if hmax value of this fact is outdated
                fact_hmax = fact_obj.hmax_value
                if fact_hmax > next_hmax:
                    # update hmax value
                    # logging.debug('updating %s' % fact_obj)
                    fact_obj.hmax_value = next_hmax
                    # enqueue all ops of which fact_obj is a hmax supporter
                    for next_op in fact_obj.precondition_of:
                        if next_op.hmax_origin == fact_obj:
                            next_op.hmax_value = next_hmax + next_op.cost
                            for supp in next_op.preconditions:
                                if supp.hmax_value + next_op.cost > next_op.hmax_value:
                                    next_op.hmax_origin = supp
                                    next_op.hmax_value = supp.hmax_value + next_op.cost
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
                    self._compute_goal_plateau(op.hmax_origin.name)


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
        if goal_state.hmax_value == float("inf"):
            return float("inf")
        while goal_state.hmax_value != 0:
            # next find an appropriate cut
            # first calculate the goal plateau
            self.goal_plateau.clear()
            self._compute_goal_plateau("GOAL")
            # then find the cut itself
            cut = self.find_cut(state)
            all_cuts.append(cut)
            # finally update heuristic value
            min_cost = min([o.cost for o in cut])
            # logging.debug("compute cut done")
            heuristic_value += min_cost
            for o in cut:
                o.cost -= min_cost
            # compute next hmax
            self._compute_hmax_from_last_cut(state, cut)
        all_cuts_names = []
        for cut in all_cuts:
            all_cuts_names.append([operator.name for operator in cut])
        if self.dead_end:
            return float("inf"), all_cuts_names
        else:
            return heuristic_value, all_cuts_names


if __name__ == "__main__":
    domain_file = '../problems/deterministic_blocksworld/domain.pddl'
    problem_file = '../problems/deterministic_blocksworld/pb3.pddl'

    parser: Parser = Parser()
    parser.scan_tokens(domain_file)
    parser.scan_tokens(problem_file)
    parser.parse_domain(domain_file)
    parser.parse_problem(problem_file)

    lmcut = LMCutHeuristic(parser)
    print(lmcut.heuristic_value)
    print(lmcut.all_cuts)