import os

from asnet.training_helper import TrainingHelper


def deterministic_blocksworld_coverage(learned_weights_path: str, min_size: int=3, max_size: int=20):
    layer_num: int = 2
    net_name: str = learned_weights_path.split('/')[-1].strip('.json')
    domain_path: str = 'problems/deterministic_blocksworld/'
    domain_file: str = domain_path + 'domain.pddl'
    problems = [f for f in os.listdir(domain_path) if os.path.isfile(os.path.join(domain_path, f)) and 'domain' not in f and 'generate' not in f]

    if 'layers' in net_name:
        layer_num = int(net_name.split('_')[-1].split('layers')[0])

    for prob in problems:
        prob_size: int = int(prob.split('_')[0].strip('pb'))
        if prob_size >= min_size and prob_size <= max_size:
            print('\t', prob)
            helper = TrainingHelper(domain_file, domain_path + prob, solve=False, asnet_layers=layer_num)
            helper.set_model_weights_from_file(learned_weights_path)

            states, _ = helper.run_policy(helper.init_state, max_steps=200)
            with open(f'results/{net_name}', 'a') as results_file:
                if helper.is_goal(states[-1]):
                    results_file.write(f'Solved {prob}\n')
                else:
                    results_file.write(f'Failed {prob}\n')
            helper = None
            states = None


def blocksworld_coverage(learned_weights_path: str, min_size: int=3, max_size: int=20):
    layer_num: int = 2
    net_name: str = learned_weights_path.split('/')[-1]
    domain_path: str = 'problems/blocksworld/'
    domain_file: str = domain_path + 'domain.pddl'
    problems = [f for f in os.listdir(domain_path) if os.path.isfile(os.path.join(domain_path, f)) and 'domain' not in f and 'generate' not in f]
    problems.sort()

    if 'layers' in net_name:
        layer_num = int(net_name.split('_')[-1].split('layers')[0])

    for prob in problems:
        prob_size: int = int(prob.split('_')[0].strip('pb'))
        if prob_size >= min_size and prob_size <= max_size:

            print('\t', prob)
            helper = TrainingHelper(domain_file, domain_path + prob, solve=False, asnet_layers=layer_num)
            helper.set_model_weights_from_file(learned_weights_path)

            states, _ = helper.run_policy(helper.init_state, max_steps=200)
            with open(f'results/{net_name}', 'a') as results_file:
                if helper.is_goal(states[-1]):
                    results_file.write(f'Solved {prob}\n')
                else:
                    results_file.write(f'Failed {prob}\n')
            helper = None
            states = None


def ip_blocksworld_coverage(learned_weights_path: str, min_size: int=3, max_size: int=20):
    layer_num: int = 2
    net_name: str = learned_weights_path.split('/')[-1]
    domain_path: str = 'problems/ip_blocksworld/'
    domain_file: str = domain_path + 'domain.pddl'
    problems = [f for f in os.listdir(domain_path) if os.path.isfile(os.path.join(domain_path, f)) and 'domain' not in f and 'generate' not in f]

    if 'layers' in net_name:
        layer_num = int(net_name.split('_')[-1].split('layers')[0])
    
    for prob in problems:
        prob_size: int = int(prob.split('_')[0].strip('pb'))
        if prob_size >= min_size and prob_size <= max_size:
            print('\t', prob)
            helper = TrainingHelper(domain_file, domain_path + prob, solve=False, asnet_layers=layer_num)
            helper.set_model_weights_from_file(learned_weights_path)

            states, _ = helper.run_policy(helper.init_state, max_steps=200)
            with open(f'results/{net_name}', 'a') as results_file:
                if helper.is_goal(states[-1]):
                    results_file.write(f'Solved {prob}\n')
                else:
                    results_file.write(f'Failed {prob}\n')
            helper = None
            states = None


def tireworld_coverage(learned_weights_path: str):
    net_name: str = learned_weights_path.split('/')[-1]
    domain_path: str = 'problems/triangle_tireworld/'
    domain_file: str = domain_path + 'domain.pddl'
    problems = [f for f in os.listdir(domain_path) if os.path.isfile(os.path.join(domain_path, f)) and 'domain' not in f and 'generate' not in f]

    for prob in problems:
        print('\t', prob)
        helper = TrainingHelper(domain_file, domain_path + prob, solve=False)
        helper.set_model_weights_from_file(learned_weights_path)

        states, _ = helper.run_policy(helper.init_state, max_steps=200)
        with open(f'results/{net_name}', 'a') as results_file:
            if helper.is_goal(states[-1]):
                results_file.write(f'Solved {prob}\n')
            else:
                results_file.write(f'Failed {prob}\n')
    
    domain_file: str = domain_path + 'domain_ip.pddl'
    for prob in problems:
        print('\t', prob)
        helper = TrainingHelper(domain_file, domain_path + prob, solve=False)
        helper.set_model_weights_from_file(learned_weights_path)

        states, _ = helper.run_policy(helper.init_state, max_steps=200)
        with open(f'results/{net_name}', 'a') as results_file:
            if helper.is_goal(states[-1]):
                results_file.write(f'Solved {prob}\n')
            else:
                results_file.write(f'Failed {prob}\n')


if __name__ == "__main__":
    base_path = "data/"
    files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
    files.remove('.gitkeep')

    files.remove("bw_0_20.json")
    files.remove("bw_20_40.json")
    files.remove("bw_3each.json")
    files.remove("bw_3each_3layers.json")
    files.remove("bw_4easy.json")
    files.remove("bw_4easy_3layers.json")
    files.remove("bw_4easy_4layers.json")

    files.remove("bw_det_0_20.json")
    files.remove("bw_det_3each.json")
    files.remove("bw_det_20_40.json")
    files.remove("bw_det_4easy.json")

    files.remove("bw_ip_0_20.json")

    print(files)

    print('bw_3each_3layers.json')
    blocksworld_coverage(base_path + 'bw_3each_3layers.json', min_size=20)
    blocksworld_coverage(base_path + 'bw_3each_3layers.json', max_size=9)

    print('bw_ip_0_20.json')
    ip_blocksworld_coverage(base_path + 'bw_det_4easy.json', min_size=18)
    ip_blocksworld_coverage(base_path + 'bw_det_4easy.json', max_size=9)

    for f in files:
        print(f)
        if 'bw' in f:
            # Blocksworld network
            if 'det' in f:
                # Deterministic network
                deterministic_blocksworld_coverage(base_path + f)
            elif 'ip' in f:
                ip_blocksworld_coverage(base_path + f)
            else:
                blocksworld_coverage(base_path + f)
        elif 'tt' in f:
            tireworld_coverage(base_path + f)
