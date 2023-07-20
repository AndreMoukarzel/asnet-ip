from typing import List, Dict, Tuple

import tensorflow as tf
from ippddl_parser.parser import Parser
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Concatenate
from tensorflow.keras.models import Model

from relations import groundify_predicate, get_related_propositions



class ASNet:

    def __init__(self, domain_file: str, problem_file: str) -> None:
        self.domain: str = domain_file
        self.problem: str = problem_file
        self.parser: Parser = Parser()
        self.parser.scan_tokens(domain_file)
        self.parser.scan_tokens(problem_file)
        self.parser.parse_domain(domain_file)
        self.parser.parse_problem(problem_file)

        # Lists lifted actions and propositions
        act_relations, pred_relations = self.get_relations(self._get_ground_actions())
        self.ground_actions = [act for act in act_relations.keys()]
        self.propositions = [pred for pred in pred_relations.keys()]
        # List to index actions and propositions by their positions
        self.act_indexed_relations = self._values_to_index(pred_relations, self.ground_actions)
        self.pred_indexed_relations = self._values_to_index(act_relations, self.propositions)

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
        """Given an unindexed dictionary with keys and values, returns an
        indexed dictionary with the indexes of the original values in the
        indexing list.
        """
        indexed_dict = {}
        for key, values in uninedexed_dict.items():
            indexed_values = [indexing_list.index(val) for val in values]
            indexed_values.sort()
            indexed_dict[key] = indexed_values
        return indexed_dict


    def _builds_connections_layer(self, layer, connection_indexes: List[int], name: str) -> Lambda:
        """Returns an intermediary Lambda layer that receives all inputs from the
        specified layer and outputs only the values of specified action_indexes.

        Essentially serves as a mask to filter only outputs from desired actions.
        """
        return Lambda(lambda x: tf.gather(x, connection_indexes, axis=1), name=name)(layer)


    def _build_input_layer(self, actions, pred_indexed_relations):
        """Builds the specially formated input action layer
        """
        input_len: int = 0
        action_sizes: Dict[str, int] = {}
        for act in actions:
            # Considers a value for each related proposition
            related_prop_num: int = len(pred_indexed_relations[act])
            # For each related proposition, indicates if it is in a goal state
            goal_info: int = related_prop_num
            # Adds one more element indicating if the action is applicable
            input_action_size: int = related_prop_num + goal_info + 1

            action_sizes[act] = input_action_size
            input_len += input_action_size
        return Input(shape=(input_len,), name="Input"), action_sizes


    def _build_first_propositions_layer(self, input_layer, input_action_sizes, propositions, act_indexed_relations, actions):
        """Builds the proposition layer that connects to the input. Since the
        input has a special format, this is slightly different than other
        proposition layers.
        """
        propositions_layer = []
        for prep in propositions:
            related_actions_indexes: List[int] = act_indexed_relations[prep]
            transformed_indexes: List[int] = []
            for act_index in related_actions_indexes:
                act_name: str = actions[act_index]
                act_input_size: int = input_action_sizes[act_name]
                # Finds the index of input action by summing the length of all
                # previous input actions
                real_act_index: int = sum([input_action_sizes[actions[act_i]] for act_i in range(act_index)])
                for i in range(real_act_index, real_act_index + act_input_size):
                    transformed_indexes.append(i)
            prop_neuron = Dense(1, name=f"{'_'.join(prep)}_1")(
                self._builds_connections_layer(input_layer, transformed_indexes, name=f"lambda_{'_'.join(prep)}_1")
            )
            propositions_layer.append(prop_neuron)
        # Concatenate all proposition neurons into a single layer
        prop_layer = Concatenate(name=f"Prop1")(propositions_layer)
        return prop_layer


    def _build_propositions_layer(self, prev_layer, propositions, act_indexed_relations, layer_num: int) -> Concatenate:
        """Builds a proposition layer.
        """
        propositions_layer = []
        for prop in propositions:
            related_actions_indexes = act_indexed_relations[prop]
            prop_neuron = Dense(1, name=f"{'_'.join(prop)}_{layer_num}")(
                self._builds_connections_layer(prev_layer, related_actions_indexes, name=f"lambda_{'_'.join(prop)}_{layer_num}")
            ) # Only connects the proposition neuron to related actions
            propositions_layer.append(prop_neuron)
        # Concatenate all proposition neurons into a single layer
        prop_layer = Concatenate(name=f"Prop{layer_num}")(propositions_layer)
        return prop_layer


    def _build_actions_layer(self, prev_layer, actions, pred_indexed_relations, layer_num: int) -> Concatenate:
        """Builds an action layer.
        """
        actions_layer = []
        for act in actions:
            act_name: str = act[0] + '_' + '_'.join(act[1])
            related_prep_indexes = pred_indexed_relations[act]
            act_neuron = Dense(1, name=f"{act_name}_{layer_num}")(
                self._builds_connections_layer(prev_layer, related_prep_indexes, name=f"lambda_{act_name}_{layer_num}")
            ) # Only connects the proposition neuron to related actions
            actions_layer.append(act_neuron)
        act_layer = Concatenate(name=f"Acts{layer_num}")(actions_layer)
        return act_layer


    def _instance_network(self, layer_num: int=2) -> Model:
        """Instances and return an Action Schema Network based on current domain
        and problem.
        
        The network will have layer_num proposition layers and (layer_num + 1)
        action layers.
        """
        input_layer, input_action_sizes = self._build_input_layer(self.ground_actions, self.pred_indexed_relations)

        last_prop_layer = self._build_first_propositions_layer(input_layer, input_action_sizes, self.propositions, self.act_indexed_relations, self.ground_actions)

        last_act_layer = self._build_actions_layer(last_prop_layer, self.ground_actions, self.pred_indexed_relations, 1)

        for i in range(2, layer_num + 1):
           last_prop_layer = self._build_propositions_layer(
               last_act_layer, self.propositions, self.act_indexed_relations, i
           )
           last_act_layer = self._build_actions_layer(
               last_prop_layer, self.ground_actions, self.pred_indexed_relations, i
           )

        return Model(
            input_layer, Dense(len(self.ground_actions), activation=tf.nn.softmax, name=f"Out_{layer_num + 1}")(last_act_layer)
        )
    

    def compile(self) -> None:
        """Compiles the ASNet to be trained"""
        self.model.compile(
            loss=keras.losses.categorical_crossentropy, # Loss function is logloss
            optimizer=keras.optimizers.SGD(learning_rate=0.0005),
            metrics=[
                'accuracy',
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                'MeanSquaredError',
                'AUC'
            ]
        )
    

    def get_model(self):
        """Returns the instanced Neural Network"""
        return self.model


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



if __name__ == "__main__":
    domain = '../problems/deterministic_blocksworld/domain.pddl'
    problem = '../problems/deterministic_blocksworld/pb1.pddl'

    asnet = ASNet(domain, problem)
    keras.utils.plot_model(asnet.model, "asnet.png", show_shapes=True)