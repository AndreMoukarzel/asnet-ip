from typing import List

from asnet import ASNet


class Trainer:

    def __init__(self, domain_file: str, problem_file: str) -> None:
        self.net = ASNet(domain_file, problem_file)
        self.parser = self.net.parser
        self.init_state = self.parser.state

        # Actual Action instances referenced by the names used in ground_actions.
        # Used when checking if an action is applicable in a state
        self.action_objects = {}
        for act in self.net._get_ground_actions():
            self.action_objects[(act.name, act.parameters)] = act
    

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



if __name__ == "__main__":
    domain = '../problems/deterministic_blocksworld/domain.pddl'
    problem = '../problems/deterministic_blocksworld/pb1.pddl'

    trainer = Trainer(domain, problem)
    print(trainer._state_to_input(trainer.init_state))