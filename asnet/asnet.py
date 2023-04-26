from typing import List, Set

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Dense, concatenate, Add
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


def get_related_action_indexes(preposition: str, relations: List[Set[str]]) -> List[int]:
    """Returns a list of action indexes of all actions to which the preposition
    is related to.

    A preposition is related to an action if it is either a precondition or an
    effect of such action.
    """
    related_actions: List[int] = []
    for action_index, rel_preps in enumerate(relations):
        if preposition in rel_preps:
            related_actions.append(action_index)
    return related_actions


def get_related_preposition_indexes(related_prepositions: Set[str], prepositions: List[str]) -> List[int]:
    """Returns a list of preposition indexes of all prepositions to which the
    action is related to.
    """
    related_prep_indexes: List[int] = []
    for prep in related_prepositions:
        prep_index = prepositions.index(prep)
        related_prep_indexes.append(prep_index)
    return related_prep_indexes


def get_actions_from_layer(layer, action_indexes: List[int]) -> Lambda:
    """Returns an intermediary Lambda layer that receives all inputs from the
    specified layer and outputs only the values of specified action_indexes.

    Essentially serves as a mask to filter only outputs from desired actions.
    """
    return Lambda(lambda x: tf.gather(x, action_indexes, axis=1))(layer)


def build_prepositions_layer(prev_layer, prepositions, relations, layer_num):
    """Builds a preposition layer.
    """
    prepositions_layer = []
    for prep in prepositions:
        related_actions_indexes = get_related_action_indexes(prep, relations)
        prep_neuron = Dense(1, name=f"{prep}{layer_num}")(
            get_actions_from_layer(prev_layer, related_actions_indexes)
        ) # Only connects the preposition neuron to related actions
        prepositions_layer.append(prep_neuron)
    prep_layer = Dense(len(prepositions), activation='relu', name=f"Prep{layer_num}")(concatenate(prepositions_layer))
    #prep_layer = Add(name=f"Prep{layer_num}")(prepositions_layer)
    return prep_layer


def build_actions_layer(prev_layer, actions, relations, prepositions, layer_num):
    """Builds an action layer.
    """
    actions_layer = []
    for act in actions:
        action_index: int = actions.index(act)
        related_prep_indexes = get_related_preposition_indexes(relations[action_index], prepositions)
        act_neuron = Dense(1, name=f"{act}{layer_num}")(
            get_actions_from_layer(prev_layer, related_prep_indexes)
        ) # Only connects the preposition neuron to related actions
        actions_layer.append(act_neuron)
    act_layer = Dense(len(actions), activation='relu', name=f"Acts{layer_num}")(concatenate(actions_layer))
    #act_layer = Add(name=f"Acts{layer_num}")(actions_layer)
    return act_layer


def create_asnet(actions: set, prepositions: set, relations: set):
    actions = list(actions)
    prepositions = list(prepositions)

    inputs = keras.Input(shape=(len(actions),), name="A1")

    prep1 = build_prepositions_layer(inputs, prepositions, relations, 1)

    act1 = build_actions_layer(prep1, actions, relations, prepositions, 1)

    prep2 = build_prepositions_layer(act1, prepositions, relations, 2)

    act2 = build_actions_layer(prep2, actions, relations, prepositions, 2)

    #out = Dense(len(prepositions))(prep1)

    return keras.Model(inputs, act2)


model = create_asnet(actions, prepositions, relations)


keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)