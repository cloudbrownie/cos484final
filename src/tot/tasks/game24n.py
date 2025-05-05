import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import * 
import random

from fractions import Fraction
from itertools import combinations, permutations
from sympy import sympify

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

OPS = [
    ('+', lambda a, b: a + b, False),                 # (symbol, fn, symmetric?)
    ('-', lambda a, b: a - b, True),                  # order matters
    ('*', lambda a, b: a * b, False),
    ('/', lambda a, b: a / b if b != 0 else None, True)
]

def solve_unique(nums, target=24, tol=0):
    """All distinct solutions for 24-game (deduped by commutativity)."""
    goal = Fraction(target)
    tol  = Fraction(tol)

    out, seen = [], set()            # seen holds canonicalised keys

    def search(vals, exprs):
        if len(vals) == 1:
            if abs(vals[0] - goal) <= tol:
                expr = exprs[0]      # keep full parentheses
                key  = str(sympify(expr, evaluate=False))   # canonical form
                if key not in seen:
                    seen.add(key)
                    out.append(expr)
            return

        for i, j in combinations(range(len(vals)), 2):
            a, b   = vals[i], vals[j]
            ea, eb = exprs[i], exprs[j]
            rest_v = [v for k, v in enumerate(vals)  if k not in (i, j)]
            rest_e = [e for k, e in enumerate(exprs) if k not in (i, j)]

            for sym, fn, ordered in OPS:
                # (+,*) are already treated as unordered by the single branch below
                for x, y, ex, ey in ( (a, b, ea, eb), ) if not ordered else \
                                     ( (a, b, ea, eb), (b, a, eb, ea) ):
                    r = fn(x, y)
                    if r is None:
                        continue
                    search(rest_v + [r], rest_e + [f'({ex} {sym} {ey})'])

    for perm in permutations(nums):
        search([Fraction(n) for n in perm], [str(n) for n in perm])


    cleaned = set()
    for soln in out:
        # Strip outermost parentheses for readability
        soln_clean = soln[1:-1] if soln.startswith("(") and soln.endswith(")") else soln
        # Remove redundant parentheses
        soln_clean = re.sub(r'\(\s*([^\(\)]+)\s*\)', r'\1', soln_clean)
        # Remove redundant spaces
        soln_clean = re.sub(r'\s+', ' ', soln_clean)
        # Remove leading/trailing spaces
        soln_clean = soln_clean.strip()
        # Add to cleaned set
        cleaned.add(soln_clean)

    out = sorted(cleaned, key=lambda x: (len(x), x))  # sort by length, then lexicographically
    return out

# ---------------------------------------------------------------------------
# Difficulty analyser  (1 = trivial … 5 = very hard)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Operation‑weighted difficulty estimator
# ---------------------------------------------------------------------------
import math
from collections import deque
from sympy import sympify, Integer, Rational, Pow, Mul, preorder_traversal

# ---------- helpers --------------------------------------------------------
def _cost_of_solution(expr_string):
    """
    Return a numeric cost for one expression.
    Lower is easier. 4nums.com's “very easy” sets tend to be <= 6.
    """
    tree = sympify(expr_string, evaluate=False)
    cost = depth = 0
    stack = [(tree, 0)]
    while stack:
        node, d = stack.pop()
        depth = max(depth, d)
        for child in node.args:
            stack.append((child, d + 1))

        # classify node -----------------------------------------------------
        if node.is_Add:
            cost += len(node.args) - 1          # n‑ary add → (n‑1) plus ops
        elif node.is_Mul:
            # division shows up as Pow(den, -1)
            for arg in node.args[1:]:
                if (isinstance(arg, Pow) and
                        arg.exp.is_Number and arg.exp.is_negative):
                    den = arg.base
                    # was it an exact division?
                    cost += 3 if den.is_Integer else 4
                else:
                    cost += 1                    # plain multiplication
        elif node.is_Pow and node.exp.is_Number and node.exp.is_negative:
            den = node.base
            cost += 3 if den.is_Integer else 4
        elif node.is_Number and isinstance(node, Rational) and node.q != 1:
            cost += 0.5                         # intermediate fraction

    # mild penalty for deep nestings beyond 3
    if depth > 3:
        cost += depth - 3
    return cost

def _search_effort(nums, target=24, cap=20000):
    """How many BFS states until first hit (log-scaled, capped)."""
    goal = Fraction(target)
    start = tuple(sorted(Fraction(n) for n in nums))
    queue = deque([start])
    seen  = {start}
    steps = 0
    while queue and steps < cap:
        state = queue.popleft(); steps += 1
        if len(state) == 1 and state[0] == goal:
            return math.log10(steps + 1)        # log scale
        for i, j in combinations(range(len(state)), 2):
            a, b = state[i], state[j]
            rest = [state[k] for k in range(len(state)) if k not in (i, j)]
            for _, fn, ordered in OPS:
                for x, y in ((a, b),) if not ordered else ((a, b), (b, a)):
                    r = fn(x, y)
                    if r is None:
                        continue
                    nxt = tuple(sorted(rest + [r]))
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
    return math.log10(cap)                      # hit the cap → harder

# ---------- main analyser ---------------------------------------------------
def analyse_task(nums) -> int:
    """
    Difficulty band 1-5 tuned toward 4nums.com intuition.
    """
    solutions = solve_unique(nums)
    if not solutions:
        return 5                                # impossible (very hard)

    # pick the cheapest solution
    #best_cost = min(_cost_of_solution(s) for s in solutions)
    import numpy as np
    costs = sorted(_cost_of_solution(s) for s in solutions)
    q25   = np.percentile(costs, 25)       # cost below which 25 % of solutions lie
    best_cost = q25

    # combine with (damped) search effort
    effort    = _search_effort(nums) * 0.6      # weight 0.6

    score = best_cost + effort

    # map score → band
    return (
        1 if score <= 6
        else 2 if score <= 8
        else 3 if score <= 10
        else 4 if score <= 12
        else 5
    )

class Game24TaskN(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, n: int = 4):
        """
        n: number of numbers to use in the game
        """
        super().__init__()
        self.data = ...
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value
    
if __name__ == "__main__":
    # to keep the generated data deterministic, we can set a seed
    random.seed(42)
    
    # print csv header
    print("rank,task,estimated difficulty")
    
    # generate 1362 random combinations of numbers from 1 to 13
    numbers = list(range(1, 14))
    n = 3
    valid = 0
    data_size = 500
    task_data = []
    
    seen = set()
    
    num_easy = 0
    num_med = 0
    
    while valid < data_size:
        # generate a random combination of n numbers and ensure it can be solved
        data = ' '.join(map(str, random.sample(numbers, n)))
        if data in seen:
            continue
        seen.add(data)        
        
        nums = list(map(int, data.split()))
        
        solutions = solve_unique(nums)
        if len(solutions) == 0:
            continue
        
        # solvable, deterimine difficulty
        difficulty = analyse_task(nums)
        
        if difficulty <= 2 and num_easy >= data_size // 3:
            continue
        elif difficulty <= 3 and num_med >= data_size // 3:
            continue
        
        if difficulty <= 2:
            num_easy += 1
        elif difficulty <= 3:
            num_med += 1
        valid += 1
        
        # append to the task data
        task_data.append((valid, data, difficulty))
        
    # sort the task data by difficulty
    task_data.sort(key=lambda x: x[2])
    
    # print the task data
    for rank, task, difficulty in task_data:
        print(f"{rank},{task},{difficulty}")
        
    # print number of easy and medium tasks generated
    print(f"Number of easy tasks generated: {num_easy}")
    print(f"Number of medium tasks generated: {num_med}")
