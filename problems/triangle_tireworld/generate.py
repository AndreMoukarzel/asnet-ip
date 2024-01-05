# Triangle Tireworld problem generator
import sys


def generate_problem(shortest_path: int) -> str:
    locations: list = []
    for i in range(shortest_path, 0, -1):
        for j in range(i, shortest_path + 1):
            locations.append(f'l-{i}-{j}')
    
    roads: list = []
    for i in range(1, shortest_path + 1):
        for j in range(i, shortest_path + 1):
            loc = f'l-{i}-{j}'
            forward = f'l-{i}-{j+1}'
            up = f'l-{i+1}-{j+1}'
            down = f'l-{i-1}-{j}'

            if j < shortest_path:
                if i % 2 == 1:
                    # Each odd-numbered road contains a forward path
                    roads.append(f'(road {loc} {forward})')
                if not (i % 2 == 0 and j % 2 == 1):
                    roads.append(f'(road {loc} {up})')

            if i > 1 and not (i % 2 == 1 and j % 2 == 0):
                # Connects to 'lower level' road
                roads.append(f'(road {loc} {down})')

    spares: list = [f'(spare-in l-1-{i})' for i in range(1, shortest_path + 1)]
    for i in range(3, shortest_path + 1, 2):
        for j in range(i + 1, shortest_path):
            spares.append(f'(spare-in l-{i}-{j})')

    return f"""(define (problem triangle-tire-{shortest_path})
        (:domain triangle-tire)
        (:objects {' '.join(locations)} - location)
    (:init (vehicle-at l-1-1) (not-flattire)
            {' '.join(roads)}
            {' '.join(spares)}
    )
    (:goal (vehicle-at l-1-{shortest_path}))
    )
    """


if __name__ == '__main__':
    max_size: int = 30
    if len(sys.argv) > 1:
        max_size = sys.argv[1]
    
    for shortest_path in range(3, max_size + 1, 2):
        problem = generate_problem(shortest_path)
        with open(f"pb{shortest_path}.pddl", "w") as file:
            file.write(problem)
