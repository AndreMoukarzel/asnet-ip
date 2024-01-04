

def generate_problem(comps_num: int) -> str:
    line_skip = '\n\t\t'
    comps: list = [f'comp{i}' for i in range(comps_num)]

    return f"""(define (problem sysadmin-{comps_num})
        (:domain sysadmin)
        (:objects {' '.join(comps)} - comp)
    (:init {''.join([f'{line_skip}(conn comp{i} comp{i + 1})' for i in range(comps_num - 1)])}
        (conn comp{0} comp{comps_num - 1})
    )
    (:goal (and {' '.join([f'(up comp{i})' for i in range(comps_num)])}))
    )
    """


if __name__ == '__main__':
    for comps_num in range(2, 31):
        problem = generate_problem(comps_num)
        with open(f"pb{comps_num}.pddl", "w") as file:
            file.write(problem)
