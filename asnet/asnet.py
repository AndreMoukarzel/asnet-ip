from typing import List, Set

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Concatenate
from tensorflow.keras.models import Model


# Blocksworld Domain
actions = {'pick-up', 'pick-up-from-table', 'put-on-block', 'put-down', 'pick-tower', 'put-tower-on-block', 'put-tower-down'}
prepositions = {'holding', 'emptyhand', 'on-table', 'on', 'clear'}
relations = [
    {'emptyhand', 'clear', 'on', 'holding', 'on-table'}, # pick-up
    {'emptyhand', 'clear', 'holding', 'on-table'}, # pick-up-from-table
    {'holding', 'clear', 'on', 'on-table'}, # put-on-block
    {'holding', 'clear', 'on-table', 'emptyhand'}, # put-down
    {'emptyhand', 'clear', 'on', 'holding'}, # pick-tower
    {'holding', 'on', 'clear', 'emptyhand', 'on-table'}, # put-tower-on-block
    {'holding', 'on', 'on-table', 'emptyhand'} # put-tower-down
]


def values_to_index(uninedexed_dict: dict, indexing_list: list):
    indexed_dict = {}
    for key, values in uninedexed_dict.items():
        indexed_values = [indexing_list.index(val) for val in values]
        indexed_values.sort()
        indexed_dict[key] = indexed_values
    return indexed_dict


def get_actions_from_layer(layer, action_indexes: List[int]) -> Lambda:
    """Returns an intermediary Lambda layer that receives all inputs from the
    specified layer and outputs only the values of specified action_indexes.

    Essentially serves as a mask to filter only outputs from desired actions.
    """
    return Lambda(lambda x: tf.gather(x, action_indexes, axis=1))(layer)


def build_prepositions_layer(prev_layer, prepositions, act_indexed_relations, layer_num):
    """Builds a preposition layer.
    """
    prepositions_layer = []
    for prep in prepositions:
        related_actions_indexes = act_indexed_relations[pred]
        prep_neuron = Dense(1, name=f"{prep}{layer_num}")(
            get_actions_from_layer(prev_layer, related_actions_indexes)
        ) # Only connects the preposition neuron to related actions
        prepositions_layer.append(prep_neuron)
    prep_layer = Concatenate(name=f"Prep{layer_num}")(prepositions_layer)
    return prep_layer


def build_actions_layer(prev_layer, actions, pred_indexed_relations, layer_num):
    """Builds an action layer.
    """
    actions_layer = []
    for act in actions:
        related_prep_indexes = pred_indexed_relations[act]
        act_neuron = Dense(1, name=f"{act}{layer_num}")(
            get_actions_from_layer(prev_layer, related_prep_indexes)
        ) # Only connects the preposition neuron to related actions
        actions_layer.append(act_neuron)
    act_layer = Concatenate(name=f"Acts{layer_num}")(actions_layer)
    return act_layer


def create_asnet(actions: set, prepositions: set, action_relations: dict, predicate_relations: dict):
    actions = list(actions)
    prepositions = list(prepositions)
    act_indexed_relations = values_to_index(predicate_relations, actions)
    pred_indexed_relations = values_to_index(action_relations, prepositions)
    print(act_indexed_relations)

    inputs = Input(shape=(len(actions),), name="A1")

    prep1 = build_prepositions_layer(inputs, prepositions, act_indexed_relations, 1)

    act1 = build_actions_layer(prep1, actions, pred_indexed_relations, 1)

    prep2 = build_prepositions_layer(act1, prepositions, act_indexed_relations, 2)

    act2 = build_actions_layer(prep2, actions, pred_indexed_relations, 2)

    #out = Dense(len(prepositions))(prep1)

    return keras.Model(inputs, act2)

model = create_asnet(actions, prepositions, relations)


keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)