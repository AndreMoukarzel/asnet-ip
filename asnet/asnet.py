import time
import logging
from typing import List, Tuple, Dict

import click
from tensorflow import keras
from keras.layers import Input

from .asnet_no_lmcut import ASNetNoLMCut


class ASNet(ASNetNoLMCut):

    def _build_input_layer(self, actions: List[Tuple[str]], related_pred_indexes: Dict[Tuple[str], List[int]]) -> Input:
        """Builds the specially formated input action layer.

        The input layer is composed - for each action - of proposition truth
        values, binary values indicating if such propositions are in a goal
        state, heuristic values related to LM-Cut and binary values indicating
        if the action is applicable.

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
            # We add three values related to the heuristic value of the action
            heuristic_info: int = 3
            # Adds one more element indicating if the action is applicable
            input_action_size: int = related_prop_num + goal_info + heuristic_info + 1

            action_sizes[act] = input_action_size
            input_len += input_action_size
        return Input(shape=(input_len,), name="Input"), action_sizes


@click.command()
@click.option("--domain", "-d", type=str, help="Path to the problem's domain PPDDL file.", default='problems/deterministic_blocksworld/domain.pddl')
@click.option("--problem", "-p", type=str, help="Path to a problem's instance PPDDL file.", default='problems/deterministic_blocksworld/pb3.pddl')
@click.option("--layer_num", "-l", type=int, help="Number of layers in the ASNet.", default=2)
@click.option("--image_name", "-img", type=str, help="Save path of the ASNet plot. By default does not save a plot.", default='')
@click.option("--debug", is_flag=True, help="Debug prints. Off by default.")
def execute(domain, problem, layer_num: int, image_name: str, debug: bool):
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s"
        )

    tic = time.process_time()
    asnet = ASNet(domain, problem, layer_num)
    toc = time.process_time()
    print(f"Built ASNet in {toc-tic}s")
    if image_name != '':
        keras.utils.plot_model(asnet.model, image_name, show_shapes=True)


if __name__ == "__main__":
    execute()