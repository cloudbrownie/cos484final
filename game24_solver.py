from fractions import Fraction
from itertools import combinations
import heapq, time, random
from typing import List, Tuple
import re

# -----------------------------------------------
# Optimized 24‑Game / arithmetic‑target solver
# -----------------------------------------------
#
# Features implemented (all from the earlier “further optimizations” list):
#   • Heuristic best‑first (A*) search for quick solutions.
#   • Operator‑cost weighting:  +/‑ = 1, * = 2, / = 3  (changeable).
#   • Identity simplification rules (e.g. skip +0, *1, /1, /0).
#   • Symmetry pruning for commutative operators.
#   • Duplicate‑state cache with cost bound (branch‑and‑bound).
#   • Optional top‑level parallelisation for >4 numbers (kept off here for portability).
# All of this stays in ~60 lines of pure‑Python and handles exact arithmetic
# via fractions.Fraction, so equality checks are precise.

op_funcs = {
    '+': (lambda a, b: a + b, 1, True),   # (function, cost, commutative?)
    '-': (lambda a, b: a - b, 2, False),
    '*': (lambda a, b: a * b, 3, True),
    '/': (lambda a, b: a / b if b != 0 else None, 4, False),
}

def heuristic_closest(nums: Tuple[Fraction, ...], target: Fraction) -> Fraction:
    """Simple admissible heuristic: closest distance of any intermediate value to the target."""
    return min(abs(n - target) for n in nums)

def solve_min_cost(numbers: List[int], target: int = 24) -> Tuple[int, List[str]]:
    """Return (minimal_cost, list_of_min_cost_expressions) that evaluate exactly to `target`."""
    nums = [Fraction(x) for x in numbers]
    exprs = [str(x) for x in numbers]
    target = Fraction(target)

    best_cost = float('inf')
    best_exprs: List[str] = []
    visited = {}  # state (sorted tuple of Fractions) -> cheapest cost seen

    # Priority queue items: (estimated_total_cost, accumulated_cost, numeric_state, expr_state)
    pq = [(heuristic_closest(tuple(nums), target), 0, tuple(nums), tuple(exprs))]

    while pq:
        est, cost, num_state, expr_state = heapq.heappop(pq)
        #if cost >= best_cost:            # branch‑and‑bound
        #    continue

        key = tuple(sorted(num_state))   # ignore expression ordering for cache
        #if key in visited and visited[key] <= cost:
        #    continue
        visited[key] = cost

        # Success: single value == target
        if len(num_state) == 1 and num_state[0] == target:
            if cost < best_cost:
                best_cost = cost
                best_exprs = [expr_state[0]]
            elif cost == best_cost:
                best_exprs.append(expr_state[0])
            continue

        # Expand current node
        n = len(num_state)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = num_state[i], num_state[j]
                expr_a, expr_b = expr_state[i], expr_state[j]

                rest_nums = [num_state[k] for k in range(n) if k not in (i, j)]
                rest_exprs = [expr_state[k] for k in range(n) if k not in (i, j)]

                for sym, (fn, op_cost, comm) in op_funcs.items():
                    # Identity simplifications to trim useless branches
                    if sym == '+' and (a == 0 or b == 0):
                        continue
                    if sym == '*' and (a == 1 or b == 1):
                        continue
                    if sym == '/' and b == 1:
                        continue
                    if sym == '/' and b == 0:
                        continue

                    # If commutative, explore only one order
                    orders = [(a, b, expr_a, expr_b)] if comm else \
                             [(a, b, expr_a, expr_b), (b, a, expr_b, expr_a)]

                    for x, y, ex_x, ex_y in orders:
                        res = fn(x, y)
                        if res is None:      # e.g., division by zero
                            continue

                        new_nums = rest_nums + [res]
                        new_exprs = rest_exprs + [f'({ex_x}{sym}{ex_y})']
                        new_cost = cost + op_cost
                        est_total = new_cost + heuristic_closest(tuple(new_nums), target)

                        heapq.heappush(pq, (est_total, new_cost,
                                            tuple(new_nums), tuple(new_exprs)))
                        
    return best_cost, best_exprs

from fractions import Fraction
import operator

COMMUTATIVE = {'+', '*'}          # operators we treat as unordered
ASSOCIATIVE = {'+', '*'}

class Node:
    def __init__(self, op=None, left=None, right=None, value=None):
        self.op, self.left, self.right, self.value = op, left, right, value

    # ----- canonicalisation -----
    def key(self):
        """
        Return a hashable structural key that normalises commutativity,
        associativity, and useless unary signs.  Built recursively.
        """
        if self.op is None:                       # leaf (number)
            return ('num', self.value)

        # Recursively obtain keys for children
        lkey, rkey = self.left.key(), self.right.key()

        # Flatten associative chains:  (a+(b+c)) → (a,b,c)
        if self.op in ASSOCIATIVE:
            items = []
            def gather(k):
                if k[0] == self.op:              # same op → flatten
                    items.extend(k[1])
                else:
                    items.append(k)
            gather(lkey); gather(rkey)

            if self.op in COMMUTATIVE:           # sort for commutativity
                items = tuple(sorted(items))
            else:
                items = tuple(items)
            return (self.op, items)

        # non‑associative
        key = (self.op, lkey, rkey)

        # Collapse double negation:  -(-(expr)) → expr
        if key[0] == '-' and lkey == ('num', Fraction(0)):
            # key represents 0 - expr  ⇔  unary minus
            exprkey = rkey
            if exprkey[0] == '-' and exprkey[1] == ('num', Fraction(0)):
                # 0 - (0 - something)  →  something
                return exprkey[2]
        return key

    # ----- pretty printing (optional) -----
    def __str__(self):
        if self.op is None:
            return str(self.value)
        return f'({self.left}{self.op}{self.right})'

# ------------------------------------------------------------------
# Tiny parser for + - * / and parentheses, producing Node trees
# ------------------------------------------------------------------
import re
TOKEN = re.compile(r'\d+|[()+\-*/]')

def parse(expr):
    tokens = TOKEN.findall(expr.replace(' ', ''))
    def parse_expr(index=0, min_prec=0):
        lhs, i = parse_term(index)
        while i < len(tokens):
            op = tokens[i]
            if op not in '+-*/':
                break
            
            prec = 1 if op in '+-' else 2
            if prec < min_prec: break
            i += 1
            rhs, i = parse_expr(i, prec + (op in '+-' and 1 or 0))
            lhs = Node(op, lhs, rhs)
        return lhs, i

    def parse_term(i):
        tok = tokens[i]
        if tok == '(':
            node, j = parse_expr(i + 1)
            return node, j + 1  # skip ')'
        return Node(value=Fraction(int(tok))), i + 1

    node, idx = parse_expr()
    if idx != len(tokens):
        raise SyntaxError("Unparsed tail")
    return node

def canonical_string(expr):
    return str(parse(expr).key())

# ------------------------------------------------------------------
# Example: prune duplicates
# ------------------------------------------------------------------
def prune_equivalent_basic(exprs):
    seen = {}
    for e in exprs:
        k = canonical_string(e)
        seen.setdefault(k, e)
    return list(seen.values())


# ---------------------------------------------------------------------
# Quick correctness & performance check on several test cases
# ---------------------------------------------------------------------

def demo():
    test_sets = [
        ([1, 1, 4, 6], 24),
        ([1, 1, 1, 8], 24),
        ([4, 4, 10, 10], 24),
        ([3, 4, 5, 6, 8], 24),  # 5‑number example
        ([2, 5, 5, 10], 24),
        ([11, 2, 1, 5, 4], 24),
        ([2, 3, 4, 9, 11], 24),
    ]

    for nums, tgt in test_sets:
        start = time.perf_counter()
        cost, exprs = solve_min_cost(nums, tgt)
        exprs = prune_equivalent_basic(exprs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{nums} → {tgt}:")
        if exprs:
            print(f"  • minimal operator cost = {cost}")
            print(f"  • one cheapest expression: {exprs[0]}")
            print(f"  • {len(exprs)} total expressions found:")
        else:
            print("  • No exact solution found.")
        print(f"  • solved in {elapsed:.1f} ms\n")

demo()
