import itertools
import numpy as np
from functools import partial
from tot.models import gpt

from heapq import heappush, heappop, nsmallest, heapify
from typing import List, Tuple, Dict

class AStarNode:
    def __init__(self, state: str, depth: int, f_score: float):
        self.state = state
        self.depth = depth
        self.f_score = f_score
        
    def __lt__(self, other):
        return self.f_score < other.f_score

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def is_solution(task, idx, expr: str) -> bool:
    """Check if the expression is a valid solution for the task."""
    return task.test_output(idx, expr)["r"] == 1

def pretty_print_solution(task_idx, expr: str):
    print(f'task {task_idx} solved with expression: {expr.strip()}')

def solve_astar(
    args,
    task,
    task_idx: int,
    *,
    to_print: bool = True,
):
    """
    A* search over thought sequences.
    g(n) = accumulated cost = depth   (shorter solutions preferred)
    h(n) = -value(n) from GPT         (more promising states preferred)
    We minimize f = g + h.  Lower f ⇒ higher priority.
    """
    # ---- initial state -----------------------------------------------------
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    
    beam_width = args.n_select_sample
    
    x = task.get_input(task_idx)
    frontier: List[AStarNode] = []
    heappush(
        frontier,
        AStarNode(
            state='',
            depth=0,
            f_score=0.0
        )
    )
    
    best_solutions = []
    best_f_score = float('inf')
    best_f = {}
        
    infos = []
        
    while frontier:
        node = heappop(frontier)
        state, depth = node.state, node.depth
        
        # check for early outs ------------------------------------------------
        if depth > task.steps:
            continue
       
        # if the depth exceeds the task's maximum steps, don't continue the leaf and see if it is a solution
        if is_solution(task, task_idx, state):
            best_solutions.append(state)
            break
            if len(best_solutions) >= beam_width:
                break
            
            continue        
            
        if depth == task.steps:
            continue
            
        # check if state has already been visited -----------------------------
        if state in best_f and node.f_score >= best_f[state]:
            continue
        best_f[state] = node.f_score

        
        # f_score check to keep track of best solution so far -----------------
        if node.f_score <= best_f_score:
            best_f_score = node.f_score
            best_expr = state # ensures incorrect output is not blank
        
        # generate step -------------------------------------------------------
        if args.method_generate == 'sample':
            candidates = get_samples(task, x, state, n_generate_sample=args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[depth])
        elif args.method_generate == 'propose':
            candidates = get_proposals(task, x, state)
        
        # evaluate step -------------------------------------------------------
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, candidates, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, candidates, args.n_evaluate_sample)
            
        # expand step ---------------------------------------------------------
        if args.method_select == 'sample':
            ps = np.array(values) / (sum(values) + 1e-12) # avoid div by zero
            idx = np.random.choice(len(candidates), size=min(len(candidates), args.n_select_sample), replace=False, p=ps)
            sampled_candidates = [(candidates[i], values[i]) for i in idx]
        elif args.method_select == 'greedy':
            sampled_candidates = sorted(zip(candidates, values), key=lambda x: x[1], reverse=True)[:args.n_select_sample]
            
        # add candidates to frontier ------------------------------------------
        for cand_state, v in sampled_candidates:
            f_score = (depth + 1) + (1 - v)
            
            #f_score = 4 - depth + 20 - v
            #max_depth = task.steps        # e.g. 4
            #alpha = 0.4

            #g = (depth + 1) / max_depth   # +1 because you’re computing child depth
            #h = 1 - (v / 20)

            #f_score = alpha * g + (1 - alpha) * h

            heappush(frontier, AStarNode(
                state=cand_state,
                depth=depth + 1,
                f_score=f_score
            ))
            
        just_candidates, _ = zip(*sampled_candidates) if sampled_candidates else ([], [])
        infos.append({'depth': depth, 'x': x, 'ys': state, 'new_ys': candidates, 'values': values, 'select_new_ys': just_candidates})

        # keep frontier small (beam-style)
        if len(frontier) > beam_width:
            frontier = list(nsmallest(beam_width, frontier))
            heapify(frontier)
            
        # log step ---------------------------------------------------------
        if to_print:
            sorted_candidates, sorted_values = zip(*sorted(zip(candidates, values), key=lambda x: x[1], reverse=True))
            print(f'-- new candidates --: {sorted_candidates}\n-- sol values --: {sorted_values}\n-- choices --: {sampled_candidates}\n')

    if len(best_solutions) == 0:
        best_solutions = ['']

    return best_solutions, {'steps': infos}
        

