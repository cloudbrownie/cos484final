import itertools
import numpy as np
from functools import partial
from tot.models import gpt

from heapq import heappush, heappop
from typing import List, Tuple, Dict

class HeapNode:
    def __init__(self, state: str, depth: int, thoughts: List[str], f_score: float):
        self.state = state
        self.depth = depth
        self.thoughts = thoughts
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
    beam_width: int = 5,
):
    """
    A* search over thought sequences.
    g(n) = accumulated cost = depth   (shorter solutions preferred)
    h(n) = -value(n) from GPT         (more promising states preferred)
    We minimize f = g + h.  Lower f ⇒ higher priority.
    """
    # ---- initial state -----------------------------------------------------
    start_state = task.get_input(task_idx)
    value = get_value(task, start_state, '', args.n_generate_sample, cache_value=False)
    frontier: List[Tuple[float, Dict]] = []
    heappush(frontier, HeapNode(
            state=start_state,
            depth=0,
            thoughts='',
            f_score=(1 - value)  # f = g + h
        )
    )

    visited = set()
    best_expr = None
        
    while frontier:
        node = heappop(frontier)
        state, depth, thoughts = node.state, node.depth, node.thoughts
        print(f'current state: {state.strip()} (depth={depth}, f_score={node.f_score})\nthoughts: {thoughts}')

        # Goal check ---------------------------------------------------------
        if is_solution(task, task_idx, state):
            best_expr = state
            break

        # Avoid revisiting the exact same partial prompt
        if state in visited:
            #print(f"Skipping already visited state: {state.strip()} out of {len(visited)} visited states.\nThey are: {list(visited)}")
            continue
        visited.add(state)

        # --------------------------------------------------------------------
        # 1) GPT‑Generate next‑step thoughts
        if args.method_generate == 'sample':
            candidates = get_samples(task, task.get_input(task_idx), state, n_generate_sample=args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[depth])
        elif args.method_generate == 'propose':
            candidates = get_proposals(task, task.get_input(task_idx), state)

        # 2) GPT‑Evaluate (heuristic) each candidate
        if args.method_evaluate == 'value':
            values = get_values(task, task.get_input(task_idx), candidates, args.n_evaluate_sample)
        elif args.method_evaluate == 'vote':
            values = get_votes(task, task.get_input(task_idx), candidates, args.n_evaluate_sample)

        # 3) Expand
        for cand_state, v in zip(candidates, values):
            f_score = (depth + 1) + (1 - v)  # g + h
            heappush(frontier, HeapNode(
                state=cand_state,
                depth=depth + 1,
                thoughts=thoughts + f'{cand_state}\n',
                f_score=f_score
            ))

        # Keep frontier small (beam‑style)
        if len(frontier) > beam_width:
            frontier = sorted(frontier)[:beam_width]
            
    if to_print and best_expr is not None:
        pretty_print_solution(task_idx, best_expr)
    return best_expr, {}
        

