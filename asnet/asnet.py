from typing import List, Set

import tensorflow as tf
from itertools import product
from ippddl_parser.parser import Parser
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Concatenate
from tensorflow.keras.models import Model



# É mais fácil pegarmos as relações pelas ações já grounded
# Assim também temos a vantagem de não incluir proposições impossíveis.
# 
# Para isso, basta considerarmos as proposições com as variáveis na hora de
# puxar as relações das ações, e adicionar também os argumentos das ações aos
# seus indicadores


class ASNet:


    def __init__(self, domain_file: str, problem_file: str) -> None:
        self.domain: str = domain_file
        self.problem: str = problem_file
        self.parser: Parser = Parser()
        self.parser.scan_tokens(domain_file)
        self.parser.scan_tokens(problem_file)
        self.parser.parse_domain(domain_file)
        self.parser.parse_problem(problem_file)

        self.model = self._instance_network()



    def _get_ground_actions(self) -> list:
        """Returns a list of the grounded actions of the problem instance"""
        ground_actions = []
        for action in self.parser.actions:
            for act in action.groundify(self.parser.objects, self.parser.types):
                ground_actions.append(act)
        return ground_actions
    

    def _get_propositions(self) -> list:
        """Returns a list of the 'grounded predicates' of the problem instance"""
        propositions = []
        for pred in self.parser.predicates:
            for prop in groundify_predicate(pred, self.parser.objects): #pred.groundify(self.parser.objects):
                propositions.append(prop)
        return propositions


    def _values_to_index(self, uninedexed_dict: dict, indexing_list: list):
        """
        """
        indexed_dict = {}
        for key, values in uninedexed_dict.items():
            indexed_values = [indexing_list.index(val) for val in values]
            indexed_values.sort()
            indexed_dict[key] = indexed_values
        return indexed_dict


    def _get_actions_from_layer(self, layer, action_indexes: List[int]) -> Lambda:
        """Returns an intermediary Lambda layer that receives all inputs from the
        specified layer and outputs only the values of specified action_indexes.

        Essentially serves as a mask to filter only outputs from desired actions.
        """
        return Lambda(lambda x: tf.gather(x, action_indexes, axis=1))(layer)


    def _build_propositions_layer(self, prev_layer, propositions, act_indexed_relations, layer_num):
        """Builds a proposition layer.
        """
        propositions_layer = []
        for prep in propositions:
            related_actions_indexes = act_indexed_relations[prep]
            prep_neuron = Dense(1, name=f"{'_'.join(prep)}{layer_num}")(
                self._get_actions_from_layer(prev_layer, related_actions_indexes)
            ) # Only connects the proposition neuron to related actions
            propositions_layer.append(prep_neuron)
        prep_layer = Concatenate(name=f"Prep{layer_num}")(propositions_layer)
        return prep_layer


    def _build_actions_layer(self, prev_layer, actions, pred_indexed_relations, layer_num):
        """Builds an action layer.
        """
        actions_layer = []
        for act in actions:
            act_name: str = act[0] + '_' + '_'.join(act[1])
            related_prep_indexes = pred_indexed_relations[act]
            act_neuron = Dense(1, name=f"{act_name}{layer_num}")(
                self._get_actions_from_layer(prev_layer, related_prep_indexes)
            ) # Only connects the proposition neuron to related actions
            actions_layer.append(act_neuron)
        act_layer = Concatenate(name=f"Acts{layer_num}")(actions_layer)
        return act_layer


    def _instance_network(self, layer_num: int=2) -> Model:
        """Instances and return an Action Schema Network based on current domain
        and problem.
        
        The network will have layer_num proposition layers and (layer_num + 1)
        action layers."""

        # Lists lifted actions and propositions
        act_relations, pred_relations = self.get_relations(self._get_ground_actions())
        ground_actions = [act for act in act_relations.keys()]
        propositions = [pred for pred in pred_relations.keys()]
        # 
        act_indexed_relations = self._values_to_index(pred_relations, ground_actions)
        pred_indexed_relations = self._values_to_index(act_relations, propositions)


        inputs = Input(shape=(len(ground_actions),), name="A1")

        prep1 = self._build_propositions_layer(inputs, propositions, act_indexed_relations, 1)

        act1 = self._build_actions_layer(prep1, ground_actions, pred_indexed_relations, 1)

        prep2 = self._build_propositions_layer(act1, propositions, act_indexed_relations, 2)

        act2 = self._build_actions_layer(prep2, ground_actions, pred_indexed_relations, 2)

        #out = Dense(len(propositions))(prep1)

        return Model(inputs, act2)


    @staticmethod
    def get_relations(actions: list) -> List[dict]:
        """Given a list of actions from an IPPDDL Parser, returns all
        propositions related to the actions and inversly all actions related to
        those propositions.
        """
        action_relations = {}
        for act in actions:
            action_relations[(act.name, act.parameters)] = get_related_propositions(act)
        
        predicate_relations = {}
        for action, related_predicates in action_relations.items():
            for pred in related_predicates:
                if pred not in predicate_relations:
                    predicate_relations[pred] = {action}
                else:
                    related_acts = predicate_relations[pred]
                    if action not in related_acts:
                        predicate_relations[pred] = set(list(related_acts) + [action])
        
        return action_relations, predicate_relations


def groundify_predicate(pred, objects):
    if not pred.arguments:
        yield pred
        return

    # For each object type of the current predicate, gets all possible objects
    all_objs = []
    for type in pred.object_types:
        if type in objects:
            all_objs.append(objects[type])
    
    # Assigns possible objects to grounded predicates and returns them
    for assignment in product(*all_objs):
        # The Predicate object doesn't accept objects with the same name,
        # which may happen after grounding, so we just return the
        # Predicate's name and its objects in order
        yield([pred.name, assignment])


def get_related_predicates(action) -> set:
        """Returns the predicates related to the action.
        """
        all_predicates = []
        for pred in action.positive_preconditions:
            all_predicates.append(pred[0])
        for pred in action.negative_preconditions:
            all_predicates.append(pred[0])
        for prop_effect in action.add_effects:
            for pred in prop_effect:
                all_predicates.append(pred[0])
        for prop_effect in action.del_effects:
            for pred in prop_effect:
                all_predicates.append(pred[0])
        return set(all_predicates)


def get_related_propositions(action) -> set:
        """Returns the propositions (predicates and their objects) related to
        the action."""
        all_predicates = []
        for pred in action.positive_preconditions:
            all_predicates.append(pred)
        for pred in action.negative_preconditions:
            all_predicates.append(pred)
        for prop_effect in action.add_effects:
            for pred in prop_effect:
                all_predicates.append(pred)
        for prop_effect in action.del_effects:
            for pred in prop_effect:
                all_predicates.append(pred)
        return set(all_predicates)



if __name__ == "__main__":
    domain = '../problems/deterministic_blocksworld/domain.pddl'
    problem = '../problems/deterministic_blocksworld/pb1.pddl'

    #asnet = ASNet(parser.action_relations, parser.predicate_relations)
    asnet = ASNet(domain, problem)
    keras.utils.plot_model(asnet.model, "asnet.png", show_shapes=True)