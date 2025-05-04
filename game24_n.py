#!/usr/bin/env python3
"""
Brute‑force “24 Game” solver for an arbitrary list length n.

Usage examples
--------------
$ python solve24.py 4 5 6 10          # find all ways to make 24
$ python solve24.py 2 3 7 11 50 -t 75 # classic "Countdown" 75‑target puzzle
"""

# TODO: determine for a set of numbers (x1 x2 x3 x4 .. xn) its "difficulty"
# base "difficulty" on the number of unique solutions (more solutions = easier)
# get_difficulty(nums: list[int | Fraction]) -> int:

from __future__ import annotations
from fractions import Fraction
from itertools import combinations, permutations
import argparse, math, sys
import operator
from sympy import sympify      # pip install sympy
from typing import List
import re

OPS = [
    ('+', lambda a, b: a + b, False),                 # (symbol, fn, symmetric?)
    ('-', lambda a, b: a - b, True),                  # order matters
    ('*', lambda a, b: a * b, False),
    ('/', lambda a, b: a / b if b != 0 else None, True)
]

def solve_unique(nums, target=24, tol=0):
    """All distinct solutions for 24‑game (deduped by commutativity)."""
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
# Verbose solver: returns a single solution plus the step‑by‑step trace
# ---------------------------------------------------------------------------

def solve_with_steps(nums, target=24):
    """
    Return a tuple (answer_expression, steps_list) or (None, None) if unsolved.

    Example
    -------
    ans, steps = solve_with_steps([2, 9, 10, 12])
    for line in steps: print(line)
    print("Answer:", ans)
    """
    goal = Fraction(target)
    nums_f = [Fraction(n) for n in nums]

    def fmt(v):
        """Pretty‑print Fractions as ints when possible."""
        return int(v) if v.denominator == 1 else v

    def search(values, exprs, trace):
        if len(values) == 1:
            return (exprs[0], trace) if values[0] == goal else (None, None)

        for i, j in combinations(range(len(values)), 2):
            a, b   = values[i], values[j]
            ea, eb = exprs[i],  exprs[j]

            rest_vals = [v for k, v in enumerate(values) if k not in (i, j)]
            rest_expr = [e for k, e in enumerate(exprs)  if k not in (i, j)]

            for sym, fn, ordered in OPS:
                # a (op) b
                res = fn(a, b)
                if res is not None:
                    new_vals  = rest_vals + [res]
                    new_exprs = rest_expr + [f"({ea} {sym} {eb})"]
                    step      = f"{fmt(a)} {sym} {fmt(b)} = {fmt(res)} (left: " + \
                                " ".join(map(str, map(fmt, new_vals))) + ")"
                    ans, log = search(new_vals, new_exprs, trace + [step])
                    if ans:
                        return ans, log

                # order‑sensitive reverse: b (op) a
                if ordered:
                    res = fn(b, a)
                    if res is not None:
                        new_vals  = rest_vals + [res]
                        new_exprs = rest_expr + [f"({eb} {sym} {ea})"]
                        step      = f"{fmt(b)} {sym} {fmt(a)} = {fmt(res)} (left: " + \
                                    " ".join(map(str, map(fmt, new_vals))) + ")"
                        ans, log = search(new_vals, new_exprs, trace + [step])
                        if ans:
                            return ans, log
        return None, None

    return search(nums_f, list(map(str, nums_f)), [])


def generate_possible_next_steps(nums):
    """
    Generate all possible next steps from the current list of numbers.
    Returns a list of strings representing the possible operations.
    """
    results = []
    nums_f = [Fraction(n) for n in nums]

    def fmt(v):
        """Pretty‑print Fractions as ints when possible."""
        return int(v) if v.denominator == 1 else v

    for i, j in combinations(range(len(nums_f)), 2):
        a, b = nums_f[i], nums_f[j]
        ea, eb = str(fmt(a)), str(fmt(b))

        for sym, fn, ordered in OPS:
            # a (op) b
            res = fn(a, b)
            if res is not None:
                step = f"{ea} {sym} {eb} = {fmt(res)} (left: " + \
                       " ".join(map(str, map(fmt, [v for k, v in enumerate(nums_f) if k not in (i, j)]))) + ")"
                results.append(step)

            # order‑sensitive reverse: b (op) a
            if ordered:
                res = fn(b, a)
                if res is not None:
                    step = f"{eb} {sym} {ea} = {fmt(res)} (left: " + \
                           " ".join(map(str, map(fmt, [v for k, v in enumerate(nums_f) if k not in (i, j)]))) + ")"
                    results.append(step)

    return results

# ---------------------------------------------------------------------------
# Convenience wrapper to pretty‑print exactly the format in your example
# ---------------------------------------------------------------------------

def show_steps(nums, target=24):
    expr, steps = solve_with_steps(nums, target)
    if not expr:
        print("No solution.")
        return
    print("Input:", *nums)
    print("Steps:")
    for s in steps:
        print(s)
    # Strip outermost parentheses for readability
    expr_clean = expr[1:-1] if expr.startswith("(") and expr.endswith(")") else expr
    print(f"Answer: {expr_clean} = {target}") 
    
_PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2}
_OP_WEIGHT  = {'+': 1.0, '-': 1.2, '*': 2.0, '/': 3.0}

def estimate_difficulty(nums):
    """Return a scalar difficulty score: lower = easier."""
    sols = solve_unique(nums)      # the 4‑nums solver you built

    def cost(expr):
        stack, ints, ops = [], 0, 0
        tokens = expr.replace('(',' ( ').replace(')',' ) ').split()
        for tok in tokens:
            if tok.isdigit():
                stack.append(int(tok))
            elif tok in '+-*/':
                b, a = stack.pop(), stack.pop()
                ops += {'+':1,'-':1.2,'*':2,'/':3}[tok]
                v = eval(f'{a}{tok}{b}')
                if v != int(v):            # not an integer
                    ops += 2
                if abs(v) > 20:            # "big" number
                    ops += math.log2(abs(v))
                stack.append(v)
        dup_bonus = -0.5 * (4 - len(set(nums)))
        return ops + dup_bonus

    return min(cost(e) for e in sols)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Brute‑force n‑number 24‑game solver.")
    parser.add_argument("numbers", type=int, nargs="+",
                        help="List of integers to use (each exactly once).")
    parser.add_argument("-t", "--target", type=int, default=24,
                        help="Target value to reach (default 24).")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Show *all* solutions (by default stop at first).")
    args = parser.parse_args(argv)

    sols = solve_unique(args.numbers, target=args.target)
    if not sols:
        print("No solution")
    else:
        print(f"Found {len(sols)} solution(s):")
        for s in sols:
            # Strip outermost parentheses for readability
            print("  ", s[1:-1] if s.startswith("(") and s.endswith(")") else s)
            
    # Print difficulty level
    difficulty = estimate_difficulty(args.numbers)
    print(f"Difficulty level: {difficulty} (1=very easy, 5=very hard)")
            
    print("Possible solution steps:")
    show_steps(args.numbers, target=args.target)

    #results = generate_possible_next_steps(args.numbers)
    #import random
    #for result in random.sample(results, 10):
    #    print(result)
    
def main2():
    from tot.tasks.game24 import Game24Task
    task = Game24Task()
    
    for task in task.data[:100]:
        # Solve the problem
        # convert input string to list of integers
        nums = list(map(int, task.split()))
        solutions = solve_unique(nums)
        
        print(f'Input: {task}, solutions: {len(solutions)}, estimated difficulty: {estimate_difficulty(nums)}')

if __name__ == "__main__":
    main()
    #main2()
