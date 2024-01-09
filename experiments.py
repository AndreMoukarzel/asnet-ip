import os

from asnet.trainer import train


def train_if_inexistent(domain, problems, layers: int=2, save: str=''):
    # Trains a new ASNet only if there is not already a saved model with the specified save name
    if not os.path.isfile(f'data/{save}.json'):
        train(domain, problems, layers=layers, save=save)
    else:
        print(f'Model {save} already trained')


def blocksworld_experiments(dir: str, problems: list, prefix: str):
    domain: str = dir + "domain.pddl"

    problems_3each = problems[:3] + problems[10:13] + problems[20:23] + problems[30:33] + problems[40:43]
    problems_4easy = [problems[0], problems[10], problems[20], problems[30]] 

    train_if_inexistent(domain, problems_4easy, save=f'{prefix}_4easy')
    train_if_inexistent(domain, problems_3each, save=f'{prefix}_3each')
    train_if_inexistent(domain, problems[:20], save=f'{prefix}_0_20')
    train_if_inexistent(domain, problems[20:40], save=f'{prefix}_20_40')
    #train_if_inexistent(domain, problems[40:60], save=f'{prefix}_40_60')
    train_if_inexistent(domain, problems_4easy, layers=3, save=f'{prefix}_4easy_3layers')
    train_if_inexistent(domain, problems_3each, layers=3, save=f'{prefix}_3each_3layers')
    train_if_inexistent(domain, problems[:20], layers=3, save=f'{prefix}_0_20_3layers')
    train_if_inexistent(domain, problems[20:40], layers=3, save=f'{prefix}_20_40_3layers')
    #train_if_inexistent(domain, problems[40:60], layers=3, save=f'{prefix}_40_60_3layers')
    #train_if_inexistent(domain, problems[40:60], layers=4, save=f'{prefix}_40_60_4layers')


def sys_admin_experiments(domain: str, prefix: str):
    problems = [f"problems/sys_admin/pb{i}.pddl" for i in range(2, 31)]
    train_if_inexistent(domain, problems[:5], save=f'{prefix}_5')
    train_if_inexistent(domain, problems[:11], save=f'{prefix}_11')
    train_if_inexistent(domain, problems[:18], save=f'{prefix}_18')
    train_if_inexistent(domain, problems[5:11], save=f'{prefix}_5_11')
    train_if_inexistent(domain, problems[11:18], save=f'{prefix}_11_18')
    train_if_inexistent(domain, problems[:5], layers=3, save=f'{prefix}_5_3layers')
    train_if_inexistent(domain, problems[:11], layers=3, save=f'{prefix}_11_3layers')
    train_if_inexistent(domain, problems[:18], layers=3, save=f'{prefix}_18_3layers')
    train_if_inexistent(domain, problems[5:11], layers=3, save=f'{prefix}_5_11_3layers')
    train_if_inexistent(domain, problems[11:18], layers=3, save=f'{prefix}_11_18_3layers')


def triangle_tireworld_experiments(domain: str, prefix: str):
    problems = [f"problems/triangle_tireworld/pb{i}.pddl" for i in range(3, 31, 2)]
    train_if_inexistent(domain, problems[:3], save=f'{prefix}_3')
    train_if_inexistent(domain, problems[:6], save=f'{prefix}_6') # pb9 could not be solved?
    train_if_inexistent(domain, problems[:9], save=f'{prefix}_9')
    train_if_inexistent(domain, problems[3:6], save=f'{prefix}_3_6')
    train_if_inexistent(domain, problems[6:9], save=f'{prefix}_6_9')
    train_if_inexistent(domain, problems[:3], layers=3, save=f'{prefix}_3_3layers')
    train_if_inexistent(domain, problems[:6], layers=3, save=f'{prefix}_6_3layers')
    train_if_inexistent(domain, problems[:9], layers=3, save=f'{prefix}_9_3layers')
    train_if_inexistent(domain, problems[3:6], layers=3, save=f'{prefix}_3_6_3layers')
    train_if_inexistent(domain, problems[6:9], layers=3, save=f'{prefix}_6_9_3layers')


if __name__ == "__main__":
    # BLOCKSWORLD EXPERIMENTS
    blocksworld_det = "problems/deterministic_blocksworld/"
    blocksworld = "problems/blocksworld/"
    blocksworld_ip = "problems/ip_blocksworld/"

    blocksworld_det_problems = []
    blocksworld_problems = []
    blocksworld_ip_problems = []
    for i in range(3, 30):
        for j in range(10):
            blocksworld_det_problems.append(blocksworld_det + f'pb{i}_p{j}.pddl')
            blocksworld_problems.append(blocksworld + f'pb{i}_p{j}.pddl')
            blocksworld_ip_problems.append(blocksworld_ip + f'pb{i}_p{j}.pddl')

    blocksworld_experiments(blocksworld_det, blocksworld_det_problems, 'bw_det')
    blocksworld_experiments(blocksworld, blocksworld_problems, 'bw')
    blocksworld_experiments(blocksworld_ip, blocksworld_ip_problems, 'bw_ip')

    # SYSADMIN EXPERIMENTS
    sys_admin_experiments("problems/sys_admin/domain.pddl", "sa")
    sys_admin_experiments("problems/sys_admin/domain_ip.pddl", "sa_ip")

    # TRIANGLE TIREWORLD EXPERIMENTS
    triangle_tireworld_experiments("problems/triangle_tireworld/domain.pddl", "tt")
    triangle_tireworld_experiments("problems/triangle_tireworld/domain_ip.pddl", "tt_ip")

    

    
