import regex as re


def _ppddl_tokenize(ppddl_txt):
    """Break PPDDL into tokens (brackets, non-bracket chunks)"""
    # strip comments
    lines = ppddl_txt.splitlines()
    mod_lines = []
    for line in lines:
        try:
            semi_idx = line.index(';')
        except ValueError:
            pass
        else:
            line = line[:semi_idx]
        mod_lines.append(line)
    ppddl_txt = '\n'.join(mod_lines)

    # convert to lower case
    ppddl_txt = ppddl_txt.lower()

    matches = re.findall(r'\(|\)|[^\s\(\)]+', ppddl_txt)

    return matches


def end_of_clause_index(tokens):
    openings = 0 # Adds 1 when '(' and subtracts 1 when ')'. If -1, then the closing was found
    for i, token in enumerate(tokens):
        if token == '(':
            openings += 1
        elif token == ')':
            openings -= 1
        
        if openings <= -1:
            return i


def get_predicates(tokens):
    for i, token in enumerate(tokens):
        if token == ":predicates":
            end_i = i + end_of_clause_index(tokens[i:])
            raw_predicates = tokens[i:end_i]
            break
    
    predicates = []
    for i, token in enumerate(raw_predicates):
        if token == '(':
            predicates.append(raw_predicates[i + 1])
    return set(predicates)


def get_actions(tokens):
    actions = []
    for i, token in enumerate(tokens):
        if token == ":action":
            actions.append(tokens[i + 1])
    return set(actions)


def get_action_relations(tokens, action: str, predicates: Set[str]):
    relations = []
    for i, token in enumerate(tokens):
        if token == action:
            end_i = i + end_of_clause_index(tokens[i:])
            action_params = tokens[i:end_i]
            break
    
    for token in action_params:
        if token in predicates:
            relations.append(token)
    
    return set(relations)