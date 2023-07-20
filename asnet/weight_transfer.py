from typing import Dict, List, Tuple

from asnet import ASNet

import numpy as np


def get_weights_by_layer(asnet: ASNet) -> List[Tuple[str, List[List[float]]]]:
    """Returns a list of tuples of format (layer name, layer type, layer weights & biases)"""
    weights_by_layer: List[Tuple[str, List[List[float]]]] = []

    for layer in asnet.model.layers:
        name: str = layer.name
        weights: List[List[float]] = layer.get_weights()
        if weights != []:
            weights_by_layer.append((name, layer.get_weights()))
    
    return weights_by_layer


def get_lifted_layer_name(grounded_layer_name: str) -> str:
    """Given a grounded layer's name, returns the name of the corresponding
    lifted layer.
    """
    layer_num: str = grounded_layer_name.split('_')[-1]
    split_name: str = grounded_layer_name.split('_')[:-1]
    if len(split_name) == 0:
        return grounded_layer_name
    lifted_name: str = split_name[0]

    if len(split_name) > 1:
        # Gives generic lifted names to objects. This way, we can
        # differentiate act-x-y from act-x-x, which will have distinct
        # weight formats
        obj_names: List[str] = split_name[1:]
        seen_objs: list = []
        for i, obj in enumerate(obj_names):
            if obj in seen_objs:
                obj_names[i] = f"obj{seen_objs.index(obj) + 1}"
            else:
                seen_objs.append(obj)
                obj_names[i] = f"obj{len(seen_objs)}"
        lifted_name += '_' + '_'.join(obj_names)
    lifted_name += '-' + layer_num

    return lifted_name


def get_lifted_weights(asnet: ASNet, pooling: str='max') -> Dict[str, List[List[float]]]:
    """Returns a dictionary of pooled weights of the trained ASNet indexed
    by lifted propositions. Those weights are appliable in other ASNets
    created in the same domain.
    """
    grounded_weights: list = get_weights_by_layer(asnet)
    lifted_weights: Dict[str, List[List[float]]] = {}

    # Lifts actions and propositions and appends all found weights for pooling
    for layer_name, weights_and_biases in grounded_weights:
        lifted_name: str = get_lifted_layer_name(layer_name)

        if lifted_name in lifted_weights:
            # If lifted object was already seen before, appends its weights for future pooling
            lifted_weights[lifted_name].append(weights_and_biases)
        else:
            lifted_weights[lifted_name] = [weights_and_biases]
    
    # Pools grounded actions' and propositions' weights into singular weight
    # values for the lifted actions and propositions
    for key, weights_and_biases in lifted_weights.items():
        if pooling == 'max':
            weights = np.array([val[0] for val in weights_and_biases])
            biases = np.array([val[1] for val in weights_and_biases])
            lifted_weights[key] = (np.max(weights, axis=0), np.max(biases, axis=0))
        elif pooling == 'mean':
            lifted_weights[key] = (np.mean(weights, axis=0), np.mean(biases, axis=0))
        else:
            raise("Unkown pooling method in get_lifted_weights()!")
    return lifted_weights


def set_lifted_weights(asnet: ASNet, lifted_weights: Dict[str, List[List[float]]]) -> None:
    """Given an instanced ASNet and a dictionary of weights lifted for the current
    domain, sets the weights on the ASNet.
    """
    for layer in asnet.model.layers:
        grounded_layer_name: str = layer.name
        lifted_name: str = get_lifted_layer_name(grounded_layer_name)

        if lifted_name in lifted_weights:
            weights_and_biases = lifted_weights[lifted_name]
            layer.set_weights(weights_and_biases)