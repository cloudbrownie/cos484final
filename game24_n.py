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
from itertools import combinations
import argparse, math, sys

OPS = [
    ('+', lambda a, b: a + b, False),                 # (symbol, fn, symmetric?)
    ('-', lambda a, b: a - b, True),                  # order matters
    ('*', lambda a, b: a * b, False),
    ('/', lambda a, b: a / b if b != 0 else None, True)
]

def solve(nums: list[int | Fraction],
          target: int | Fraction = 24,
          find_all: bool = True,
          tol: Fraction | None = None) -> list[str]:
    """
    Returns a list of expression strings that evaluate to `target`.
    If find_all=False, returns as soon as the first solution is found.
    """
    goal   = Fraction(target)
    tol    = Fraction(0) if tol is None else Fraction(tol)
    nums_f = [Fraction(n) for n in nums]

    def search(values: list[Fraction], exprs: list[str]) -> list[str]:
        if len(values) == 1:
            return [exprs[0]] if abs(values[0] - goal) <= tol else []

        results = []
        # choose unordered pair positions i < j
        for i, j in combinations(range(len(values)), 2):
            a, b = values[i], values[j]
            ea, eb = exprs[i], exprs[j]

            # generate new lists after removing positions i & j
            rest_vals = [v for k, v in enumerate(values) if k not in (i, j)]
            rest_expr = [e for k, e in enumerate(exprs)  if k not in (i, j)]

            for sym, fn, ordered in OPS:
                # try a (op) b
                res = fn(a, b)
                if res is not None:
                    new_vals = rest_vals + [res]
                    new_expr = rest_expr + [f"({ea} {sym} {eb})"]
                    found = search(new_vals, new_expr)
                    results.extend(found)
                    if found and not find_all:
                        return results

                # if the operator is order‑sensitive, also try b (op) a
                if ordered:
                    res = fn(b, a)
                    if res is not None:
                        new_vals = rest_vals + [res]
                        new_expr = rest_expr + [f"({eb} {sym} {ea})"]
                        found = search(new_vals, new_expr)
                        results.extend(found)
                        if found and not find_all:
                            return results
        return results

    start_exprs = [str(n) for n in nums_f]
    return search(nums_f, start_exprs)

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

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Brute‑force n‑number 24‑game solver.")
    parser.add_argument("numbers", type=int, nargs="+",
                        help="List of integers to use (each exactly once).")
    parser.add_argument("-t", "--target", type=int, default=24,
                        help="Target value to reach (default 24).")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Show *all* solutions (by default stop at first).")
    args = parser.parse_args(argv)

    sols = solve(args.numbers, target=args.target, find_all=args.all)
    if not sols:
        print("No solution")
    else:
        print(f"Found {len(sols)} solution(s):")
        for s in sols:
            # Strip outermost parentheses for readability
            print("  ", s[1:-1] if s.startswith("(") and s.endswith(")") else s)
            
    show_steps(args.numbers, target=args.target)

    results = generate_possible_next_steps(args.numbers)
    import random
    for result in random.sample(results, 10):
        print(result)

if __name__ == "__main__":
    main()
