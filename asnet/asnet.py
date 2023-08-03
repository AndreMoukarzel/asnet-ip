from typing import List, Dict, Tuple
import itertools

import tensorflow as tf
from ippddl_parser.parser import Parser
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Concatenate, Maximum, Reshape
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
        self.ground_actions.sort()
        self.propositions = [pred for pred in pred_relations.keys()]
        self.propositions.sort()
        # List to index actions and propositions by their positions
        self.act_indexed_relations = self._values_to_index(pred_relations, self.ground_actions)
        self.pred_indexed_relations = self._values_to_index(act_relations, self.propositions)

        self.model = self._instance_network()


    def _get_ground_actions(self) -> list:
        """Returns a list of the grounded actions of the problem instance"""
        ground_actions = []
        for action in self.parser.actions:
            for act in action.groundify(self.parser.objects, self.parser.types):
                # Does not add actions invalidated by equality preconditions
                invalid_equalty: bool = False
                for precond in act.negative_preconditions:
                    if precond[0] == 'equal' and precond[1] == precond[2]:
                        invalid_equalty = True
                
                if not invalid_equalty:
                    ground_actions.append(act)
        return ground_actions


    def _get_propositions(self) -> list:
        """Returns a list of the 'grounded predicates' of the problem instance"""
        propositions = []
        for pred in self.parser.predicates:
            for prop in groundify_predicate(pred, self.parser.objects):
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
        propositions_layer: List[Dense] = []
        for prep in propositions:
            related_actions_indexes: List[int] = act_indexed_relations[prep]
            transformed_indexes: List[int] = self.action_indexes_to_input(related_actions_indexes, actions, input_action_sizes)

            prop_neuron = Dense(1, name=f"{'_'.join(prep)}_1")(
                self._builds_connections_layer(input_layer, transformed_indexes, name=f"lambda_{'_'.join(prep)}_1")
            )
            propositions_layer.append(prop_neuron)
        # Concatenate all proposition neurons into a single layer
        prop_layer = Concatenate(name=f"Prop1")(propositions_layer)
        return prop_layer
    

    def _get_related_predicates(self, related_action_indexes) -> List[List[int]]:
        """Returns a list where each element is a list of action indexes that
        refer to the same action and have at least one predicate in a common
        position.
        (i. e. 'drive(shakey, hall, kitchen)' and 'drive(shakey, hall, office)')
        """
        related_predicates: List[List[int]] = []

        for act_index in related_action_indexes:
            action: tuple = self.ground_actions[act_index]
            act_name: str = action[0]
            act_predicates: Tuple[str] = action[1]
            related_indexes: List[int] = [act_index]
            
            for other_act_index in related_action_indexes:
                if other_act_index != act_index:
                    other_action: str = self.ground_actions[other_act_index]
                    other_predicates: Tuple[str] = other_action[1]

                    # Checks for other related actions representing the same action type
                    if act_name == other_action[0]:
                        # If both actions have a common predicate in the same position, they are related
                        for i, pred in enumerate(act_predicates):
                            if pred == other_predicates[i]:
                                related_indexes.append(other_act_index)
                                break
            
            related_indexes = list(set(related_indexes)) # Removes repetitions
            related_indexes.sort() # Sorts list for future repetition removal in related_predicates

            related_predicates.append(related_indexes)
        
        # Removes repetitions
        related_predicates.sort()
        related_predicates = list(val for val,_ in itertools.groupby(related_predicates))

        return related_predicates


    def _pool_related_predicates(self, prev_layer, related_predicates, name: str) -> Reshape:
        """Pools maximum value of propositions with related predicates into a
        a single output
        """
        lambda_layers = []
        for pred in related_predicates:
            lambda_layers.append(Lambda(lambda x: tf.gather(x, pred, axis=1))(prev_layer))
        return Reshape([1])(Maximum(name=name)(lambda_layers))


    def _build_propositions_layer(self, prev_layer, propositions, act_indexed_relations, layer_num: int) -> Concatenate:
        """Builds a proposition layer.
        """
        propositions_layer: List[Dense] = []
        lifted_prop_neurons: Dict[str, Dense] = {}
        for prop in propositions:
            lifted_prop_name: str = prop[0]
            related_actions_indexes: List[int] = act_indexed_relations[prop]
            related_predicates: List[List[int]] = self._get_related_predicates(related_actions_indexes)

            pooled_layers: list = []
            for i, preds in enumerate(related_predicates):
                # Pools related predicates into single values before adding them as input
                if len(preds) > 1:
                    pooled_layers.append(
                        self._pool_related_predicates(prev_layer, preds, name=f"pooled_{'_'.join(prop)}_{i}_{layer_num}")
                    )
            # Gathers all propositions with no relation to others into a single Lambda layer
            solo_preds: List[int] = self.unify_solo_elements(related_predicates)
            solo_lambda = self._builds_connections_layer(prev_layer, solo_preds, name=f"solo_{'_'.join(prop)}_{layer_num}")
            pooled_layers.append(solo_lambda)

            concat_pooled = Concatenate(name=f"concat_{'_'.join(prop)}_{layer_num}")(pooled_layers)

            # Creates a neuron representing a single proposition from the proposition layer
            prop_neuron = Dense(1, name=f"{'_'.join(prop)}_{layer_num}")
            prop_neuron.build(concat_pooled.shape)

            # Weight sharing between prop neurons representing the same action with different predicates
            if lifted_prop_name not in lifted_prop_neurons:
                # First time prop was seen
                lifted_prop_neurons[lifted_prop_name] = prop_neuron
            else:
                # Share weights with other prop of same type
                self.share_layer_weights(lifted_prop_neurons[lifted_prop_name], prop_neuron)
            
            prop_neuron = prop_neuron(concat_pooled)
            propositions_layer.append(prop_neuron)
        # Concatenate all proposition neurons into a single layer
        prop_layer = Concatenate(name=f"Prop{layer_num}")(propositions_layer)
        return prop_layer


    def _build_actions_layer(self, prev_layer, actions, pred_indexed_relations, layer_num: int) -> Concatenate:
        """Builds an action layer.

        All action layers representing a same action (e.g. PickUp(a) and PickUp(b)) share weights, so their weights are
        generalizable in the domain.
        """
        actions_layer: List[Dense] = []
        lifted_act_neurons: Dict[str, Dense] = {}
        for act in actions:
            act_name: str = act[0] + '_' + '_'.join(act[1])
            lifted_act_name: str = act[0]
            related_prep_indexes: List[int] = pred_indexed_relations[act]

            lambda_layer = (
                self._builds_connections_layer(prev_layer, related_prep_indexes, name=f"lambda_{act_name}_{layer_num}")
            ) # Only connects the proposition neuron to related actions
            act_neuron = Dense(1, name=f"{act_name}_{layer_num}")
            act_neuron.build(lambda_layer.shape)

            # Weight sharing between actions neurons representing the same action with different predicates
            if lifted_act_name not in lifted_act_neurons:
                # First time action was seen
                lifted_act_neurons[lifted_act_name] = act_neuron
            else:
                # Share weights with other actions of same type
                self.share_layer_weights(lifted_act_neurons[lifted_act_name], act_neuron)
            
            act_neuron = act_neuron(lambda_layer)

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
    

    @staticmethod
    def action_indexes_to_input(related_actions_indexes, actions, input_action_sizes) -> List[int]:
        """Transforms action indexes to the equivalent format found in the
        input vector composed of only boolean values representing all predicates.
        
        Returns the list of action indexes transformed into the equivalent input
        indexes.
        """
        transformed_indexes: List[int] = []

        for act_index in related_actions_indexes:
            act_name: str = actions[act_index]
            act_input_size: int = input_action_sizes[act_name]
            # Finds the index of input action by summing the length of all
            # previous input actions
            real_act_index: int = sum([input_action_sizes[actions[act_i]] for act_i in range(act_index)])
            for i in range(real_act_index, real_act_index + act_input_size):
                transformed_indexes.append(i)
        
        return transformed_indexes
    

    @staticmethod
    def unify_solo_elements(all_elements: List[list]) -> list:
        """Given a list of lists, returns the concatenation of all
        single-element lists."""
        solo_elements: list = []
        for elements in all_elements:
            if len(elements) == 1:
                solo_elements.append(elements[0])
        return solo_elements
    

    @staticmethod
    def share_layer_weights(layer1, layer2):
        layer2.kernel = layer1.kernel
        layer2.bias = layer1.bias
        layer2._trainable_weights = []
        layer2._trainable_weights.append(layer2.kernel)
        layer2._trainable_weights.append(layer2.bias)


if __name__ == "__main__":
    domain = '../problems/deterministic_blocksworld/domain.pddl'
    problem = '../problems/deterministic_blocksworld/pb3.pddl'

    asnet = ASNet(domain, problem)
    keras.utils.plot_model(asnet.model, "asnet1.jpg", show_shapes=True)