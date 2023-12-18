# BlocksWorld problem generator
from random import choice, shuffle, randint

from ippddl_parser.parser import Parser


def randomize_init_state(blocks: list) -> frozenset:
    initial_state = []
    unassigned_blocks = []
    free_positions = ['table']
    for obj in blocks:
        unassigned_blocks.append(obj)
        initial_state.append(('equal', obj, obj)) # Adds equality condition to all blocks
    
    # Assigns a random initial position to each block
    # (Can be placed on top of previously placed blocks)
    while unassigned_blocks:
        chosen_block = choice(unassigned_blocks)
        unassigned_blocks.remove(chosen_block)
        pos = choice(free_positions)
        if pos == 'table':
            initial_state.append(('on-table', chosen_block))
        else:
            initial_state.append(('on', chosen_block, pos))
            free_positions.remove(pos)
        free_positions.append(chosen_block)
    
    # For free blocks, adds the 'clear' predicate
    free_positions.remove('table')
    for block in free_positions:
        initial_state.append(('clear', block))
    
    # Adds emptyhand predicate
    initial_state.append(('emptyhand', ))

    return frozenset(initial_state)


def randomize_goal_state(init_state, actions, min_num_steps: int=10, max_num_steps: int=50) -> frozenset:
    state = init_state
    for _ in range(randint(min_num_steps, max_num_steps)):
        shuffle(actions)
        for act in actions:
            if act.is_applicable(state):
                state = act.apply(state)
                break
    return state


def pred_to_str(pred) -> str:
    # Converts state predicates to a string parseable in PDDL
    if len(pred) == 1:
        return f'({pred[0]})'
    return f"({pred[0]} {' '.join(pred[1:])})"


def problem_to_pddl(blocks, init_state, goal_state, problem_name: str) -> str:
    return f"""(define (problem {problem_name})
  (:domain blocksworld)
  (:objects {' '.join(blocks)} - block)
  (:init {' '.join([pred_to_str(pred) for pred in init_state])})
  (:goal (and {' '.join([pred_to_str(pred) for pred in goal_state])}))
)
"""

def generate_problem(block_num: int) -> str:
    domain_file: str = 'domain.pddl'
     
    parser: Parser = Parser()
    parser.scan_tokens(domain_file)
    parser.parse_domain(domain_file)

    parser.objects = {'block': [f'b{i}' for i in range(1, block_num + 1)]}
    parser.state = randomize_init_state(parser.objects['block'])

    grounded_actions: list = []
    for action in parser.actions:
        for act in action.groundify(parser.objects, parser.types):
            grounded_actions.append(act)

    goal_state = randomize_goal_state(parser.state, grounded_actions)
    return problem_to_pddl(parser.objects['block'], parser.state, goal_state, f'bw_{block_num}')


if __name__ == '__main__':
    for block_num in range(3, 20):
        for i in range(10):
            problem = generate_problem(block_num)
            with open(f"pb{block_num}_p{i}.pddl", "w") as file:
                file.write(problem)
