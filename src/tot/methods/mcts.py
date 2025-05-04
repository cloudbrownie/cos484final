import math
import random
from collections import defaultdict
import itertools
import numpy as np
from functools import partial
from tot.models import gpt

from heapq import heappush, heappop
from typing import List, Tuple, Dict

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

def solve_mcts(
    args,
    task,
    task_idx: int,
    *,
    to_print: bool = True,
    n_simulations: int = 100,
    c_puct: float = 1.2,
):
    """
    Plain UCT MCTS where:
      * Expansion uses GPT‑Generate (propose)
      * Rollout / leaf evaluation uses GPT‑Value (value)
    """
    root_state = task.get_initial_state(task_idx)
    root = task.hash_state(root_state)

    # tree statistics --------------------------------------------------------
    children = defaultdict(list)    # parent ID -> [child IDs]
    N = defaultdict(int)            # visit count
    W = defaultdict(float)          # total value
    P = {}                          # prior (value estimate) for each state

    state_cache = {root: root_state}

    # ------------------------------------------------------------------------
    def policy_and_value(s):
        """Expand: return list[(child_state, prior)] and value for leaf."""
        samples = get_samples(
            args, task, task_idx, s, n_generate_sample=args.n_generate_sample
        )
        priors = get_value(args, task, task_idx, samples)
        return samples, priors

    # ------------------------------------------------------------------------
    for _ in range(n_simulations):
        path: List[str] = [root]
        state = state_cache[root]

        # SELECTION
        while children[path[-1]]:
            parent = path[-1]
            # UCB‑style score
            total_visits = N[parent]
            def ucb(child):
                Q = W[child] / (1 + N[child])
                U = c_puct * P[parent][child] * math.sqrt(total_visits) / (1 + N[child])
                return Q + U
            best_child = max(children[parent], key=ucb)
            path.append(best_child)
            state = state_cache[best_child]

            #ys, infos, finished = finish(args, task, task_idx, state)
            #if finished:
            #    # Terminal node reached; back‑prop a perfect value
            #    leaf_value = 1.0
            #    break
            
            if is_solution(task, task_idx, state):
                # Terminal node reached; back‑prop a perfect value
                leaf_value = 1.0
                best_expr = state
                break
            
        else:
            # EXPANSION
            samples, priors = policy_and_value(state)
            if samples:
                parent = path[-1]
                child_probs = {}
                for child_state, p in zip(samples, priors):
                    cid = task.hash_state(child_state)
                    if cid not in state_cache:
                        state_cache[cid] = child_state
                    children[parent].append(cid)
                    child_probs[cid] = p
                P[parent] = child_probs

            # ROLLOUT / LEAF EVAL
            leaf_value = max(priors) if priors else 0.0

        # BACK‑PROP
        for node in reversed(path):
            N[node] += 1
            W[node] += leaf_value

    # choose best child from root
    best = max(children[root], key=lambda c: N[c])
    best_state = state_cache[best]
    #ys, infos, _ = finish(args, task, task_idx, best_state)

    #if to_print:
    #    _print_solution(task_idx, ys, infos)

    #return ys, infos

    if to_print:
        pretty_print_solution(task_idx, best_state)
    return [best_expr], {}
