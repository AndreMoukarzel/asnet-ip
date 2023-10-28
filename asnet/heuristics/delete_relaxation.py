"""Auxiliary classes used in describing a delete-relaxed problem from a problem
contained in Parser.
"""
from typing import List, Tuple, Dict

from ippddl_parser.parser import Parser


class RelaxedFact:
    """Object representing a relaxed fact and its related operators in a
    justification graph
    """
    def __init__(self, name: Tuple[str]):
        self.name: Tuple[str] = name
        self.heuristic_value: float = float("inf")
        self.precondition_of: List[RelaxedOperator] = []
        self.effect_of: List[RelaxedOperator] = []

    # Set comparison functions for a RelaxedFact
    def __lt__(self, other: "RelaxedFact"):
        return self.heuristic_value < other.heuristic_value

    def __leq__(self, other: "RelaxedFact"):
        return self.heuristic_value <= other.heuristic_value

    def __gt__(self, other: "RelaxedFact"):
        return self.heuristic_value > other.heuristic_value

    def __geq__(self, other: "RelaxedFact"):
        return self.heuristic_value >= other.heuristic_value
    
    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"Fact {self.name}, Heuristic Value: {self.heuristic_value}" + \
            f"\n \t precond of: {[str(p) for p in self.precondition_of]}" + \
            f"\n \t effect of: {[str(p) for p in self.effect_of]}"

    def clear(self):
        self.heuristic_value = float("inf")


class RelaxedOperator:
    def __init__(self, name: str, cost_zero: bool=False):
        self.name: str = name
        self.preconditions: List[RelaxedFact] = []
        self.effects: List[RelaxedFact] = []
        self.heuristic_value: float = float("inf")
        self.hval_origin: RelaxedFact = None # RelaxedFact from which the HMax was obtained, if applicable
        self.cost: float = 1.0
        self.cost_zero: bool = cost_zero
        if self.cost_zero:
            self.cost = 0.0
    

    def __lt__(self, other: "RelaxedOperator"):
        return self.heuristic_value < other.heuristic_value


    def reset_cost(self):
        self.cost = 1.0
        if self.cost_zero:
            self.cost = 0.0


class JustificationGraph:
    """Justification Graph's representation of a delete-relaxated problem"""

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
    

    @staticmethod
    def add_precondition_to_operator(precondition: RelaxedFact, operator: RelaxedOperator):
        """Adds RelaxedFact as precondition to RelaxedOperator, and vice-versa"""
        if precondition not in operator.preconditions:
            operator.preconditions.append(precondition)
        if operator not in precondition.precondition_of:
            precondition.precondition_of.append(operator)


    @staticmethod
    def add_effect_to_operator(effect: RelaxedFact, operator: RelaxedOperator):
        """Adds RelaxedFact as effect to RelaxedOperator, and vice-versa"""
        if effect not in operator.effects:
            operator.effects.append(effect)
        if operator not in effect.effect_of:
            effect.effect_of.append(operator)
