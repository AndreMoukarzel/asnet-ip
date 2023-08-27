"""Contains the ASNet class, used for instancing an Action Schema Network.

The only trainable layers of the network are its action and proposition layers.
All other layers, such as the Lambda and Concatenate layers, are used simply to
make sure the action and proposition layers have the appropriate format to have
shareable weights.
"""
from typing import List, Dict, Tuple
import itertools

from ippddl_parser.parser import Parser
from tensorflow import keras
from keras.layers import Input, Dense, Concatenate
from keras.models import Model

from .custom_layers import Output, ActionModule, PropositionModule
from .relations import groundify_predicate, get_related_propositions


DEBUG: bool = False



class ASNet:
    """
    Action Schema Network architecture.

    The ASNet architecture is defined by the grounded actions and propositions
    from the specified problem instance, but its layers' weights can be
    transfered to any other ASNet built from the same problem's domain.

    Methods
    -------
    compile()
        Compiles the ASNet for training
    get_model()

    get_relations(actions)
        Returns all propositions related to each action and all actions related
        to each propositions
    action_indexes_to_input(related_actions_indexes, actions, input_action_sizes)
        Converts action indexes to equivalent indexes from the input layer
    """

    def __init__(self, domain_file: str, problem_file: str) -> None:
        """
        Parameters
        ----------
        domain_file : str
            IPPDDL file specifying the problem domain
        problem_file : str
            IPPDDL file specifying the problem instance
        """
        self.domain: str = domain_file
        self.problem: str = problem_file
        self.parser: Parser = Parser()
        if DEBUG:
            print("Building parser")
        self.parser.scan_tokens(domain_file)
        self.parser.scan_tokens(problem_file)
        self.parser.parse_domain(domain_file)
        self.parser.parse_problem(problem_file)

        # Lists lifted actions and propositions
        act_relations, pred_relations = self.get_relations(self._get_ground_actions())
        self.ground_actions: List[Tuple[str]] = [act for act in act_relations.keys()]
        self.ground_actions.sort()
        self.propositions: List[Tuple[str]] = [pred for pred in pred_relations.keys()]
        self.propositions.sort()
        # List to index actions and propositions by their positions
        self.related_act_indexes: Dict[Tuple[str], List[int]] = self._values_to_index(pred_relations, self.ground_actions)
        self.related_pred_indexes: Dict[Tuple[str], List[int]] = self._values_to_index(act_relations, self.propositions)

        self.model = self._instance_network()


    def _get_ground_actions(self) -> list:
        """Returns a list of the grounded actions of the problem instance.

        Returned actions are instances of ippddl_parser's Action objects.
        """
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


    def _values_to_index(self, unindexed_values: Dict[any, List[str]], values_to_index: List[any]) -> Dict[any, List[int]]:
        """Converts the values from the 'uninedexed_values' dictionary into the
        index those values have in the 'values_to_index' list

        Parameters
        ----------
        unindexed_values: Dict[Tuple[str], List[str]]
            Dictionary with values to be converted to indexes
        values_to_index: List[str]
            List of values found in the dictionary above. The INDEXES of values
            in this list will be used as values in the returned dictionary
        
        Returns
        -------
        Dict[any, List[int]]
            Dictionary with keys from 'unindexed_values' and a list of indexes
            extracted from 'values_to_index' as values.
        """
        indexed_dict: dict = {}
        for key, values in unindexed_values.items():
            indexed_values = [values_to_index.index(val) for val in values]
            indexed_values.sort()
            indexed_dict[key] = indexed_values
        return indexed_dict


    def _build_input_layer(self, actions: List[Tuple[str]], related_pred_indexes: Dict[Tuple[str], List[int]]) -> Input:
        """Builds the specially formated input action layer.

        The input layer is composed - for each action - of proposition truth
        values, binary values indicating if such propositions are in a goal
        state and binary values indicating if the action is applicable.

        Parameters
        ----------
        actions: List[Tuple[str]]
            List of tuples or format (action name, action predicates)
        related_pred_indexes: Dict[Tuple[str], List[int]]
            Dictionary where the keys are action tuples such as above, and
            the values are the list of indexes of related predicates to each
            actions.
        
        Returns
        -------
        Input
            Input layer of the ASNet
        """
        input_len: int = 0
        action_sizes: Dict[str, int] = {}
        for act in actions:
            # Considers a value for each related proposition
            related_prop_num: int = len(related_pred_indexes[act])
            # For each related proposition, indicates if it is in a goal state
            goal_info: int = related_prop_num
            # Adds one more element indicating if the action is applicable
            input_action_size: int = related_prop_num + goal_info + 1

            action_sizes[act] = input_action_size
            input_len += input_action_size
        return Input(shape=(input_len,), name="Input"), action_sizes
    

    def _get_related_predicates(self, actions_indexes: List[int]) -> List[List[int]]:
        """Returns a list where each element is a list of action indexes that
        refer to the same action and have at least one predicate in a common
        position.
        (i. e. 'drive(shakey, hall, kitchen)' and 'drive(shakey, hall, office)')

        Parameters
        ----------
        actions_indexes: List[int]
            List of indexes of actions
        
        Returns
        -------
        List[List[int]]
            List with list of related actions' indexes.
        """
        related_predicates: List[List[int]] = []

        for act_index in actions_indexes:
            action: tuple = self.ground_actions[act_index]
            act_name: str = action[0]
            act_predicates: Tuple[str] = action[1]
            related_indexes: List[int] = [act_index]
            
            for other_act_index in actions_indexes:
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


    def _build_first_propositions_layer(self, input_layer: Input, input_action_sizes: List[int]):
        """Builds the proposition layer that connects to the input. Since the
        input has a special format, this is slightly different than other
        proposition layers.
        """
        layer_num: int = 1
        propositions_layer: List[Dense] = []
        lifted_prop_modules: Dict[str, Dense] = {}
        if DEBUG:
            print("First Prop Layer\n")
        for prop in self.propositions:
            if DEBUG:
                print(f"Building {prop} Layer")
            lifted_prop_name: str = prop[0]
            related_actions_indexes: List[int] = self.related_act_indexes[prop]
            if DEBUG:
                print("\tGetting related")
            related_predicates: List[List[int]] = self._get_related_predicates(related_actions_indexes)
            if DEBUG:
                print("\tDone getting related")
            
            related_connections: List[List[int]] = only_grouped_elements(related_predicates)
            transformed_related_connections: List[List[int]] = [self.action_indexes_to_input(conn, self.ground_actions, input_action_sizes) for conn in related_connections]
            # Gathers all propositions with no relation to others into a single Lambda layer
            unrelated_connections: List[int] = unify_solo_elements(related_predicates)
            transformed_unrelated_connections: List[int] = self.action_indexes_to_input(unrelated_connections, self.ground_actions, input_action_sizes)

            prop_module = PropositionModule(transformed_related_connections, transformed_unrelated_connections, name=f"{'_'.join(prop)}_{layer_num}")
            prop_module.build_weights()

            # Weight sharing between prop neurons representing the same action with different predicates
            if lifted_prop_name not in lifted_prop_modules:
                # First time prop was seen
                lifted_prop_modules[lifted_prop_name] = prop_module
            else:
                # Share weights with other prop of same type
                share_layer_weights(lifted_prop_modules[lifted_prop_name], prop_module)

            propositions_layer.append(prop_module(input_layer))
        # Concatenate all proposition neurons into a single layer
        prop_layer = Concatenate(name=f"Prop{layer_num}", trainable=False)(propositions_layer)
        return prop_layer


    def _build_propositions_layer(self, prev_layer, layer_num: int) -> Concatenate:
        """Builds a proposition layer.
        """
        propositions_layer: List[Dense] = []
        lifted_prop_modules: Dict[str, Dense] = {}
        if DEBUG:
            print(f"Prop Layer {layer_num}\n")
        for prop in self.propositions:
            if DEBUG:
                print(f"Building {prop} Layer")
            lifted_prop_name: str = prop[0]
            related_actions_indexes: List[int] = self.related_act_indexes[prop]
            if DEBUG:
                print("\tGetting related")
            related_predicates: List[List[int]] = self._get_related_predicates(related_actions_indexes)
            if DEBUG:
                print("\tDone getting related")

            related_connections: List[List[int]] = only_grouped_elements(related_predicates)
            unrelated_connections: List[int] = unify_solo_elements(related_predicates)

            prop_module = PropositionModule(related_connections, unrelated_connections, name=f"{'_'.join(prop)}_{layer_num}")
            prop_module.build_weights()

            # Weight sharing between prop neurons representing the same predicate
            if lifted_prop_name not in lifted_prop_modules:
                # First time prop was seen
                lifted_prop_modules[lifted_prop_name] = prop_module
            else:
                # Share weights with other prop of same type
                share_layer_weights(lifted_prop_modules[lifted_prop_name], prop_module)
            
            propositions_layer.append(prop_module(prev_layer))
                
        # Concatenate all proposition neurons into a single layer
        prop_layer = Concatenate(name=f"Prop{layer_num}", trainable=False)(propositions_layer)
        return prop_layer


    def _build_actions_layer(self, prev_layer, layer_num: int) -> Concatenate:
        """Builds an action layer.

        All action layers representing a same action (e.g. PickUp(a) and PickUp(b))
        share weights, so their weights are generalizable in the domain.
        """
        actions_layer: List[Dense] = []
        lifted_act_neurons: Dict[str, Dense] = {}
        if DEBUG:
            print(f"Action Layer {layer_num}\n")
        for act in self.ground_actions:
            act_name: str = act[0] + '_' + '_'.join(act[1])
            lifted_act_name: str = act[0]
            related_prep_indexes: List[int] = self.related_pred_indexes[act]
            act_neuron = ActionModule(related_prep_indexes, name=f"{act_name}_{layer_num}")
            act_neuron.build_weights()

            # Weight sharing between actions neurons representing the same action with different predicates
            if lifted_act_name not in lifted_act_neurons:
                # First time action was seen
                lifted_act_neurons[lifted_act_name] = act_neuron
            else:
                # Share weights with other actions of same type
                share_layer_weights(lifted_act_neurons[lifted_act_name], act_neuron)

            actions_layer.append(act_neuron(prev_layer))
        act_layer = Concatenate(name=f"Acts{layer_num}", trainable=False)(actions_layer)
        return act_layer


    def _instance_network(self, layer_num: int=2) -> Model:
        """Instances and return an Action Schema Network based on current domain
        and problem.
        
        The network will have layer_num proposition layers and (layer_num + 1)
        action layers.
        """
        input_layer, input_action_sizes = self._build_input_layer(self.ground_actions, self.related_pred_indexes)
        last_prop_layer = self._build_first_propositions_layer(input_layer, input_action_sizes)
        last_act_layer = self._build_actions_layer(last_prop_layer, 1)

        for i in range(2, layer_num + 1):
           last_prop_layer = self._build_propositions_layer(last_act_layer, i)
           last_act_layer = self._build_actions_layer(last_prop_layer, i)

        if DEBUG:
            print("Building output layer")
        
        output_layer = Output(
            input_action_sizes, trainable=False, name="Out"
        )([last_act_layer, input_layer])

        return Model(input_layer, output_layer)
    

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


########## HELPER METHODS ##########

def unify_solo_elements(all_elements: List[list]) -> list:
    """Given a list of lists, returns the concatenation of all
    single-element lists."""
    solo_elements: list = []
    for elements in all_elements:
        if len(elements) == 1:
            solo_elements.append(elements[0])
    return solo_elements


def only_grouped_elements(all_elements: List[list]) -> List[List[int]]:
    """Given a list of lists, returns all elements with length larger than 1"""
    grouped_elements: List[List[int]] = []
    for elements in all_elements:
        if len(elements) > 1:
            grouped_elements.append(elements)
    return grouped_elements


def share_layer_weights(layer1, layer2):
    kernel, bias = layer1.get_trainable_weights()
    layer2.set_trainable_weights(kernel, bias)


if __name__ == "__main__":
    domain = 'problems/deterministic_blocksworld/domain.pddl'
    problem = 'problems/deterministic_blocksworld/pb3.pddl'

    asnet = ASNet(domain, problem)
    #keras.utils.plot_model(asnet.model, "asnet.jpg", show_shapes=True)