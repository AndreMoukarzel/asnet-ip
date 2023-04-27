from ppddl_parser import _ppddl_tokenize, get_actions, get_predicates, get_predicate_relations, get_action_relations
from asnet.asnet import create_asnet

import keras


if __name__ == "__main__":
    # Blocksworld Domain
    with open('problems/blocksworld/domain.pddl', 'r') as fp:
        pddl_txt = fp.read()
        tokens = _ppddl_tokenize(pddl_txt)

    actions = get_actions(tokens)
    predicates = get_predicates(tokens)
    action_relations = {}
    predicate_relations = {}

    for act in actions:
        action_relations[act] = get_action_relations(tokens, act, get_predicates(tokens))
    for pred in predicates:
        predicate_relations[pred] = get_predicate_relations(pred, action_relations)

    print(actions)
    print(predicates)
    print(action_relations)
    print(predicate_relations)

    model = create_asnet(actions, predicates, action_relations, predicate_relations)

    keras.utils.plot_model(model, "model_shape.png", show_shapes=True)