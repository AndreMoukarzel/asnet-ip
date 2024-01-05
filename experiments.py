from asnet.trainer import train


def blocksworld_experiments(dir: str, problems: list, prefix: str):
    domain: str = dir + "domain.pddl"

    problems_3each = problems[:3] + problems[10:13] + problems[20:23] + problems[30:33] + problems[40:43]

    train(domain, problems_3each, save=f'{prefix}_3each')
    train(domain, problems[:40], save=f'{prefix}_0_40')
    train(domain, problems[20:40], save=f'{prefix}_20_40')
    train(domain, problems[:60], save=f'{prefix}_0_60')
    train(domain, problems[40:60], save=f'{prefix}_40_60')
    train(domain, problems[40:60], layers=3, save=f'{prefix}_40_60_3layers')
    train(domain, problems[40:60], layers=4, save=f'{prefix}_40_60_4layers')


def sys_admin_experiments(domain: str, prefix: str):
    problems = [f"problems/sys_admin/pb{i}.pddl" for i in range(2, 31)]
    train(domain, problems[:5], save=f'{prefix}_5')
    train(domain, problems[:11], save=f'{prefix}_11')
    train(domain, problems[:18], save=f'{prefix}_18')
    train(domain, problems[5:11], save=f'{prefix}_5_11')
    train(domain, problems[11:18], save=f'{prefix}_11_18')
    train(domain, problems[:5], layers=3, save=f'{prefix}_5_3layers')
    train(domain, problems[:11], layers=3, save=f'{prefix}_11_3layers')
    train(domain, problems[:18], layers=3, save=f'{prefix}_18_3layers')
    train(domain, problems[5:11], layers=3, save=f'{prefix}_5_11_3layers')
    train(domain, problems[11:18], layers=3, save=f'{prefix}_11_18_3layers')


def triangle_tireworld_experiments(domain: str, prefix: str):
    problems = [f"problems/triangle_tireworld/pb{i}.pddl" for i in range(3, 31, 2)]
    train(domain, problems[:3], save=f'{prefix}_3')
    train(domain, problems[:6], save=f'{prefix}_6')
    train(domain, problems[:9], save=f'{prefix}_9')
    train(domain, problems[3:6], save=f'{prefix}_3_6')
    train(domain, problems[6:9], save=f'{prefix}_6_9')
    train(domain, problems[:3], layers=3, save=f'{prefix}_3_3layers')
    train(domain, problems[:6], layers=3, save=f'{prefix}_6_3layers')
    train(domain, problems[:9], layers=3, save=f'{prefix}_9_3layers')
    train(domain, problems[3:6], layers=3, save=f'{prefix}_3_6_3layers')
    train(domain, problems[6:9], layers=3, save=f'{prefix}_6_9_3layers')


if __name__ == "__main__":
    # BLOCKSWORLD EXPERIMENTS
    blocksworld_det = "problems/deterministic_blocksworld/"
    blocksworld = "problems/blocksworld/"
    blocksworld_ip = "problems/ip_blocksworld/"

    blocksworld_det_problems = []
    blocksworld_problems = []
    blocksworld_ip_problems = []
    for i in range(3, 40):
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
    sys_admin_experiments("problems/triangle_tireworld/domain.pddl", "tt")
    sys_admin_experiments("problems/triangle_tireworld/domain_ip.pddl", "tt_ip")
