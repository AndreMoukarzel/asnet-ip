from typing import List, Set

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Concatenate
from tensorflow.keras.models import Model



class ASNet:

    def __init__(self, action_relations: dict, predicate_relations: dict) -> None:
        self.model = self._create_asnet(action_relations, predicate_relations)


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


    def _build_prepositions_layer(self, prev_layer, prepositions, act_indexed_relations, layer_num):
        """Builds a preposition layer.
        """
        prepositions_layer = []
        for prep in prepositions:
            related_actions_indexes = act_indexed_relations[prep]
            prep_neuron = Dense(1, name=f"{prep}{layer_num}")(
                self._get_actions_from_layer(prev_layer, related_actions_indexes)
            ) # Only connects the preposition neuron to related actions
            prepositions_layer.append(prep_neuron)
        prep_layer = Concatenate(name=f"Prep{layer_num}")(prepositions_layer)
        return prep_layer


    def _build_actions_layer(self, prev_layer, actions, pred_indexed_relations, layer_num):
        """Builds an action layer.
        """
        actions_layer = []
        for act in actions:
            related_prep_indexes = pred_indexed_relations[act]
            act_neuron = Dense(1, name=f"{act}{layer_num}")(
                self._get_actions_from_layer(prev_layer, related_prep_indexes)
            ) # Only connects the preposition neuron to related actions
            actions_layer.append(act_neuron)
        act_layer = Concatenate(name=f"Acts{layer_num}")(actions_layer)
        return act_layer


    def _create_asnet(self, action_relations: dict, predicate_relations: dict):
        actions = [act for act in action_relations.keys()]
        prepositions = [pred for pred in predicate_relations.keys()]
        act_indexed_relations = self._values_to_index(predicate_relations, actions)
        pred_indexed_relations = self._values_to_index(action_relations, prepositions)

        inputs = Input(shape=(len(actions),), name="A1")

        prep1 = self._build_prepositions_layer(inputs, prepositions, act_indexed_relations, 1)

        act1 = self._build_actions_layer(prep1, actions, pred_indexed_relations, 1)

        prep2 = self._build_prepositions_layer(act1, prepositions, act_indexed_relations, 2)

        act2 = self._build_actions_layer(prep2, actions, pred_indexed_relations, 2)

        #out = Dense(len(prepositions))(prep1)

        return Model(inputs, act2)


if __name__ == "__main__":
    from ippddl_parser.parser import Parser

    domain = '../problems/deterministic_blocksworld/domain.pddl'
    problem = '../problems/deterministic_blocksworld/pb1.pddl'
    parser = Parser()
    parser.scan_tokens(domain)
    parser.scan_tokens(problem)
    parser.parse_domain(domain)
    parser.parse_problem(problem)
    asnet = ASNet(parser.action_relations, parser.predicate_relations)

    keras.utils.plot_model(asnet.model, "asnet.png", show_shapes=True)