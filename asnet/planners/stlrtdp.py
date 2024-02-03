from .lrtdp import LRTDP, is_goal


class STLRTDP(LRTDP):    
    def _bellman_update(self, s):
        '''
        Adapts the traditional Bellman Update method to follow the minmax
        criteria, where we assume that all actions will result in the future
        state that offers the minimum reward.

        As seen in page 54 of
        https://teses.usp.br/teses/disponiveis/45/45134/tde-15022010-161012/publico/dissertacao.pdf
        '''
        app_acts = self._get_applicable_actions(s)
        self.state_values[s] = max(self.min_Q(s, a) for a in app_acts)


    def min_Q(self, s, a):
        sucessors, _ = self._get_successor_states(s)
        if sucessors is None:
            # If there are no successor states (aka the state is absorbing) we consider the state solved.
            return 0
        min_q = 999.0
        future_states, probs = a.get_possible_resulting_states(s)
        for i, ns in enumerate(future_states):
            reward: float = 0.0
            if self._is_goal(ns):
                reward = 1.0
            q_val = probs[i] * (reward + self.discount_rate * self.state_values[ns])
            min_q = min(min_q, q_val)
        return min_q
    

    def _check_solved(self, s):
        # The difference from LRTDP is using max(self.min_Q(s, a)) instead of Q(pi)
        flag = True
        open = []
        closed = []
        if not self.solved_states[s]:
            open.append(s)
        while open:
            s = open.pop()
            closed.append(s)
            #app_acts = self._get_applicable_actions(s)
            residual = self.state_values[s] - self.min_Q(s, self.policy(s))# for a in app_acts)
            if abs(residual) > self.bellman_error_margin:
                flag = False
            else:
                successor_states, _ = self._get_successor_states(s)
                for ns in successor_states:
                    if not self.solved_states[ns] and ns not in open and ns not in closed:
                        open.append(ns)
        if flag:
            for ns in closed:
                self.solved_states[ns] = True
        else:
            while closed:
                s = closed.pop()
                self._bellman_update(s)
        return flag


if __name__ == "__main__":
    from ippddl_parser.parser import Parser
    from ..heuristics.lm_cut import LMCutHeuristic

    domain_file = 'problems/ip_blocksworld/domain.pddl'
    problem_file = 'problems/ip_blocksworld/pb4_p1.pddl'

    parser: Parser = Parser()
    parser.scan_tokens(domain_file)
    parser.scan_tokens(problem_file)
    parser.parse_domain(domain_file)
    parser.parse_problem(problem_file)

    lmcut = LMCutHeuristic(parser)

    lrtdp = STLRTDP(parser, lmcut)
    lrtdp.execute()

    s = parser.state
    print(s)
    for _ in range(20):
        if is_goal(s, parser.positive_goals, parser.negative_goals):
            print("GOAL REACHED")
            break
        act = lrtdp.policy(s)
        s = act.apply(s)
        print(f'|-> {act.name}[{act.parameters}] -> {s}')
