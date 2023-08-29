from typing import Dict, List, Tuple

from .asnet import ASNet

import numpy as np


def get_weights_by_layer(asnet: ASNet) -> List[Tuple[str, np.array]]:
    """Returns a list of tuples of format (layer name, layer's weights & biases)
    
    Given an ASNet instance, for all layers that have weights and biases, which
    should correspond to all and only the ActionModules and PropositionModules,
    returs a list with their weights.

    Parameters
    ----------
    asnet: ASNet
        ASNet instance with built neural network.
    
    Returns
    -------
    List[Tuple[str, np.array]]
        List of tuples with format (Layer name, layer's weights & biases)
    """
    weights_by_layer: List[Tuple[str, np.array]] = []

    for layer in asnet.model.layers:
        name: str = layer.name
        weights: np.array = layer.get_weights()
        if weights != []:
            weights_by_layer.append((name, layer.get_weights()))
    
    return weights_by_layer


def get_lifted_name(grounded_name: str) -> str:
    """Given the name of a grounded action or proposition, returns the name of
    such object when lifted.

    Names are expected to be the action/proposition name, followed by the
    related predicates and at last the layer number, all separated by underlines.
    
    The name 'hold_block1_2' would reference the action Hold(block1), and be
    extracted from an ActionModule from layer 2 of an ASNet.

    Parameters
    ----------
    grounded_name: str
        Name of the object with grounded predicates.

    Returns
    -------
    str
        Lifted name. E.g. the propositions on_a_b_1 and on_b_c_1 would both
        be transformed and returned as on_obj1_obj2.
    """
    layer_num: str = grounded_name.split('_')[-1]
    split_name: str = grounded_name.split('_')[:-1]
    if len(split_name) == 0:
        return grounded_name
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


def get_lifted_weights(asnet: ASNet, pooling: str='max') -> Dict[str, np.array]:
    """Returns a dictionary of pooled weights of the trained ASNet indexed
    by name of Action/Proposition.

    The names of ActionModules or PropositionModules the weights are originated
    from are lifted, meaning it won't be differentiated, in the returned values,
    if a weight came from the proposition ('clear', 'a') or ('clear', 'b').

    Traditionally, weights of all objects with the same lifted name would be
    the same in an ASNet, but this function takes in consideration the
    possibility of them being different, and pools such weights according to the
    specified pooling method.

    Parameters
    ----------
    asnet: ASNet
        ASNet instance from which the weights will be extracted from.
    pooling: str, optional
        Pooling method to aggregate weights from objects with equal LIFTED names.
        Accepts 'max' and 'mean'/'avg'. Default is 'max'.
    
    Returns
    -------
    Dict[str, np.array]
        Dicitonary where the key is the lifted name of an action or proposition
        and the value are its weights and biases.
    """
    grounded_weights: list = get_weights_by_layer(asnet)
    lifted_weights: Dict[str, np.array] = {}

    # Lifts actions and propositions and appends all found weights for pooling
    for layer_name, weights_and_biases in grounded_weights:
        lifted_name: str = get_lifted_name(layer_name)
        
        if lifted_name in lifted_weights:
            # If lifted object was already seen before, appends its weights for future pooling
            lifted_weights[lifted_name].append(weights_and_biases)
        else:
            lifted_weights[lifted_name] = [weights_and_biases]
    
    # Pools grounded actions' and propositions' weights into singular weight
    # values for the lifted actions and propositions
    for key, weights_and_biases in lifted_weights.items():
        weights = np.array([val[0] for val in weights_and_biases])
        biases = np.array([val[1] for val in weights_and_biases])
        if pooling == 'max':
            lifted_weights[key] = (np.max(weights, axis=0), np.max(biases, axis=0))
        elif pooling == 'mean' or pooling == 'avg':
            lifted_weights[key] = (np.mean(weights, axis=0), np.mean(biases, axis=0))
        else:
            raise("Unkown pooling method in get_lifted_weights()!")
    return lifted_weights


def set_lifted_weights(asnet: ASNet, lifted_weights: Dict[str, np.array]) -> None:
    """Given an instanced ASNet and a dictionary of weights for the current
    domain's lifted actions and propositions, sets the weights on the ASNet.

    Parameters
    ----------
    asnet: ASNet
        ASNet instance which will have its weights overwritten.
    lifted_weights: Dict[str, np.array]
        Dicitonary where the key is the lifted name of an action or proposition
        and the value are its weights and biases. Can be extracted from an
        ASNet using the get_lifted_weights() function.
    """
    for layer in asnet.model.layers:
        grounded_layer_name: str = layer.name
        lifted_name: str = get_lifted_name(grounded_layer_name)

        if lifted_name in lifted_weights:
            weights_and_biases = lifted_weights[lifted_name]
            layer.set_weights(weights_and_biases)