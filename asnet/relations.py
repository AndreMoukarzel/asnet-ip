from itertools import product


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