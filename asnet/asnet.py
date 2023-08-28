"""Contains the ASNet class, used for instancing an Action Schema Network.

The only trainable layers of the network are its action and proposition layers.
All other layers, such as the Lambda and Concatenate layers, are used simply to
make sure the action and proposition layers have the appropriate format to have
shareable weights.
"""
import logging
from typing import List, Dict, Tuple
import itertools

from ippddl_parser.parser import Parser
from tensorflow import keras
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
import click

from .custom_layers import Output, ActionModule, PropositionModule
from .relations import get_related_propositions



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
        logging.info(f"Building ASNet based on:\n\tDomain: {domain_file}\n\tProblem: {problem_file}")
        self.domain: str = domain_file
        self.problem: str = problem_file
        self.parser: Parser = Parser()

        logging.debug("Building parser")
        self.parser.scan_tokens(domain_file)
        self.parser.scan_tokens(problem_file)
        self.parser.parse_domain(domain_file)
        self.parser.parse_problem(problem_file)

        # Lists actions and propositions
        # The keys are sorted so their relative order is the same in all ASNets instanced from the same problem domain.
        # The keys are tuples of strings representing the action/proposition and their related predicates, respectively
        # Example Action keys: ('PickUp', ('a',)) | ('Stack', ('b', 'c'))
        # Example Propoposition Keys: ('clear', 'a') | ('on', 'c', 'b')
        act_relations, pred_relations = self.get_relations(self._get_ground_actions())
        self.ground_actions: List[Tuple[str]] = [act for act in act_relations.keys()]
        self.ground_actions.sort()
        self.propositions: List[Tuple[str]] = [pred for pred in pred_relations.keys()]
        self.propositions.sort()
        # List actions related to each proposition by their index in self.ground_actions
        self.related_act_indexes: Dict[Tuple[str], List[int]] = self._values_to_index(pred_relations, self.ground_actions)
        # List propositions related to each action by their index in self.propositions
        self.related_prop_indexes: Dict[Tuple[str], List[int]] = self._values_to_index(act_relations, self.propositions)

        self.model = self._instance_network()


    ################################################ PRIVATE METHODS ################################################


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


    def _build_first_propositions_layer(self, input_layer: Input, input_action_sizes: List[int]) -> Concatenate:
        """Builds the proposition layer that connects to the Input Layer from an
        ASNet.
        
        Since the input has a special format, this is slightly different than
        other proposition layers.

        Parameters
        ----------
        input_layer: Input
            Input Layer of the ASNet
        input_action_sizes: List[int]
            Size of each action input in the Input Layer. Since the actions in
            the Input Layer are represented by a vector based on their related
            predicates, they may have variable sizes.
        
        Returns
        -------
        Concatenate
            Concatenation of multiple PropositionModule instances forming a
            Proposition Layer.
        """
        layer_num: int = 1
        propositions_layer: List[Dense] = []
        lifted_prop_modules: Dict[str, Dense] = {}
        logging.debug("First Prop Layer\n")
        for prop in self.propositions:
            logging.debug(f"Building {prop} 's PropositionModule")
            lifted_prop_name: str = prop[0]
            related_actions_indexes: List[int] = self.related_act_indexes[prop]
            logging.debug("\tGetting related predicates")
            related_predicates: List[List[int]] = self._get_related_predicates(related_actions_indexes)
            logging.debug("\tDone getting related")
            
            related_connections: List[List[int]] = get_grouped_elements(related_predicates)
            transformed_related_connections: List[List[int]] = [self.action_indexes_to_input(conn, self.ground_actions, input_action_sizes) for conn in related_connections]
            # Gathers all propositions with no relation to others into a single Lambda layer
            unrelated_connections: List[int] = get_solo_elements(related_predicates)
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


    def _build_propositions_layer(self, prev_layer: Concatenate, layer_num: int) -> Concatenate:
        """Builds a proposition layer by concatenation of multiple
        PropositionModule instances.

        All proposition layers representing the same proposition (e.g. Clear(a)
        and Clear(b)) share weights. Given that their weights have the same
        format, they can also be generalized to any equivalent proposition
        layers in other ASNets based on the same problem domain.

        Parameters
        ----------
        prev_layer: Concatenate
            Previous Action Layer.
        layer_num: int
            Number of the current layer.
        
        Returns
        -------
        Concatenate
            Concatenation of multiple PropositionModule instances forming a
            Proposition Layer.
        """
        propositions_layer: List[Dense] = []
        lifted_prop_modules: Dict[str, Dense] = {}
        logging.debug(f"Prop Layer {layer_num}\n")
        for prop in self.propositions:
            logging.debug(f"Building {prop} 's PropositionModule")
            lifted_prop_name: str = prop[0]
            related_actions_indexes: List[int] = self.related_act_indexes[prop]
            logging.debug("\tGetting related predicates")
            related_predicates: List[List[int]] = self._get_related_predicates(related_actions_indexes)
            logging.debug("\tDone getting related")

            related_connections: List[List[int]] = get_grouped_elements(related_predicates)
            unrelated_connections: List[int] = get_solo_elements(related_predicates)

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
        """Builds an action layer by concatenation of multiple ActionModule
        instances.

        All action layers representing the same action (e.g. PickUp(a) and
        PickUp(b)) share weights. Given that their weights have the same
        format, they can also be generalized to any equivalent action layers in
        other ASNets based on the same problem domain.

        Parameters
        ----------
        prev_layer: Concatenate
            Previous Proposition Layer.
        layer_num: int
            Number of the current layer.
        
        Returns
        -------
        Concatenate
            Concatenation of multiple ActionModule instances forming a, Action
            Layer.
        """
        actions_layer: List[Dense] = []
        lifted_act_neurons: Dict[str, Dense] = {}
        logging.debug(f"Action Layer {layer_num}")
        for act in self.ground_actions:
            act_name: str = act[0] + '_' + '_'.join(act[1])
            lifted_act_name: str = act[0]
            related_prop_indexes: List[int] = self.related_prop_indexes[act]
            logging.debug(f"Building {act_name} 's ActionModule")
            act_neuron = ActionModule(related_prop_indexes, name=f"{act_name}_{layer_num}")
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
        
        The network will have layer_num Proposition Layers and (layer_num + 1)
        Action Layers.

        Parameters
        ----------
        layer_num: int
            Number of Action and Proposition Layers to be built in the Network.
        
        Raises
        ------
        ValueError
            If the layer has an invalid (lower than 1) number of layers.
        
        Returns
        -------
        Model
            The ASNet network.
        """
        if layer_num < 1:
            raise ValueError('The network must have at least one layer. The value of layer_num must be equal or larger than 1.')
        input_layer, input_action_sizes = self._build_input_layer(self.ground_actions, self.related_prop_indexes)
        last_prop_layer = self._build_first_propositions_layer(input_layer, input_action_sizes)
        last_act_layer = self._build_actions_layer(last_prop_layer, 1)

        for i in range(2, layer_num + 1):
           last_prop_layer = self._build_propositions_layer(last_act_layer, i)
           last_act_layer = self._build_actions_layer(last_prop_layer, i)

        logging.debug("Building output layer")
        output_layer = Output(
            input_action_sizes, trainable=False, name="Out"
        )([last_act_layer, input_layer])
        logging.debug("Done building ASNet")

        return Model(input_layer, output_layer)
    

    ################################################ PUBLIC METHODS ################################################
    

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
    

    def get_model(self) -> Model:
        """Returns the instanced Neural Network"""
        return self.model


    ################################################ STATIC METHODS ################################################


    @staticmethod
    def get_relations(actions: list) -> List[Dict[Tuple[str], Tuple[str]]]:
        """Given a list of actions from an IPPDDL Parser, returns all
        propositions related to the actions and inversly all actions related to
        those propositions.

        Parameters
        ----------
        actions: List[ippddl_parser.Action]
            List of actions
        
        Returns
        -------
        List[Dict[Tuple[str], Tuple[str]]]
            Returns a dictionary where keys are the actions and values are the
            related propositions, as well as a dictionary were keys are the
            propositions and values are the related actions.
        """
        action_relations: Dict[Tuple[str], Tuple[str]] = {}
        for act in actions:
            action_relations[(act.name, act.parameters)] = get_related_propositions(act)
        
        predicate_relations: Dict[Tuple[str], Tuple[str]] = {}
        # Uses the built action_relations dictionary to infer the reversed relationships 
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
    def action_indexes_to_input(action_indexes: List[int], actions: List[Tuple[str]], input_action_sizes: List[int]) -> List[int]:
        """Returns a list of the Input Layer's indexes that is equivalent to
        the actions from the received action_indexes in any other Action Layer
        of the ASNet.

        Parameters
        ----------
        action_indexes: List[int]
            List of indexes of relevant actions.
        actions: List[Tuple[str]]
            List of action names.
        input_action_sizes: List[int]
            List with size of each action in the Input Layer of the ASNet.
        """
        transformed_indexes: List[int] = []

        for act_index in action_indexes:
            act_name: str = actions[act_index]
            act_input_size: int = input_action_sizes[act_name]
            # Finds the index of input action by summing the length of all
            # previous input actions
            real_act_index: int = sum([input_action_sizes[actions[act_i]] for act_i in range(act_index)])
            for i in range(real_act_index, real_act_index + act_input_size):
                transformed_indexes.append(i)
        
        return transformed_indexes


#################################################### HELPER METHODS ####################################################


def get_solo_elements(all_elements: List[list]) -> list:
    """Given a list of lists, returns the concatenation of all
    single-element lists."""
    solo_elements: list = []
    for elements in all_elements:
        if len(elements) == 1:
            solo_elements.append(elements[0])
    return solo_elements


def get_grouped_elements(all_elements: List[list]) -> List[List[int]]:
    """Given a list of lists, returns all elements with length larger than 1"""
    grouped_elements: List[List[int]] = []
    for elements in all_elements:
        if len(elements) > 1:
            grouped_elements.append(elements)
    return grouped_elements


def share_layer_weights(layer1, layer2) -> None:
    """Shares weights of layer1 with layer2 structurally.

    In other words, layer2 will now train the same weights that layer1, even if
    such layers are not directly connected.

    Parameters
    ----------
    layer1: ActionModule or PropositionModule
        Layer from which the original weights will be extracted
    layer2: ActionModule or PropositionModule
        Layer to receive new weights and biases from layer1.
    """
    kernel, bias = layer1.get_trainable_weights()
    layer2.set_trainable_weights(kernel, bias)



@click.command()
@click.option("--domain", "-d", type=str, help="Path to the problem's domain PPDDL file.", default='problems/deterministic_blocksworld/domain.pddl')
@click.option("--problem", "-p", type=str, help="Path to a problem's instance PPDDL file.", default='problems/deterministic_blocksworld/pb3.pddl')
@click.option("--image_name", "-img", type=str, help="Save path of the ASNet plot. By default does not save a plot.", default='')
@click.option("--debug", is_flag=True, help="Debug prints. Off by default.")
def execute(domain, problem, image_name: str, debug: bool):
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s"
        )

    asnet = ASNet(domain, problem)
    if image_name != '':
        keras.utils.plot_model(asnet.model, image_name, show_shapes=True)


if __name__ == "__main__":
    execute()