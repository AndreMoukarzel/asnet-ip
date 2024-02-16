from itertools import product
from typing import Set, List, Tuple


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


def get_related_propositions(action, include_equality: bool=False) -> Set[Tuple[str]]:
    """Returns all propositions (predicates and their objects) related to
    the action.
    
    Parameters
    ----------
    action: ippddl_parser.Action
        Action instance with its defined preconditions and effects.

    Returns
    -------
    List[Tuple[str]]
        All propositions related to the action represented as tuples
        of format (proposition_name, object1, object2, ...).

        E.g. ('on', 'a', 'b')
    """
    all_predicates: List[Tuple[str]] = []
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
    
    if include_equality:
        return all_predicates

    # Removes equality predicates. Since they are static, they are not relevant
    # to the network's training.
    cleaned_predicates = []
    for pred in all_predicates:
        if not 'equal' in pred[0]:
            cleaned_predicates.append(pred)

    return set(cleaned_predicates) # Removes repetitions
