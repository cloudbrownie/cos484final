import argparse
from tot.methods.bfs import solve
from tot.methods.astar import solve_astar
from tot.methods.mcts import solve_mcts
from tot.tasks.game24 import Game24Task
import random
import asyncio
from functools import partial

args = argparse.Namespace(
    backend='gpt-4.1-mini', 
    temperature=0.7, 
    task='game24', 
    naive_run=False, 
    prompt_sample=None, 
    method_generate='propose', 
    method_evaluate='value', 
    method_select='greedy', 
    n_generate_sample=1, 
    n_evaluate_sample=3, 
    n_select_sample=5,
    solve_method='astar'  # Change this to 'astar' or 'mcts' as needed
)

task = Game24Task()

idx_min = 901
idx_max = 1000

# sample random numbers from range without replacement
samples = 20
tasks = random.sample(range(idx_min, idx_max + 1), samples)

# limit concurrent threads so you stay under OpenAI’s parallel‑request quota
CONCURRENCY = 5
sema = asyncio.Semaphore(CONCURRENCY)


if args.solve_method == 'bfs':
    solve_func = solve
elif args.solve_method == 'astar':
    solve_func = partial(solve_astar, beam_width=5)
elif args.solve_method == 'mcts':
    solve_func = partial(solve_mcts, n_simulations=20)

# ---- thin wrappers --------------------------------------------------------
def _solve_sync(task_idx: int):
    """Run the original blocking solve() in a worker thread."""
    print(f"Solving task {task_idx}: {task.get_input(task_idx)}")
    #ys, _ = solve(args, task, task_idx, to_print=False)
    ys, _ = solve_func(args, task, task_idx, to_print=False)
    ok = task.test_output(task_idx, ys)["r"] == 1
    print(f"Task {task_idx}: {'Correct' if ok else 'Incorrect'} - {ys}")
    return task_idx, ok, ys

async def solve_one(task_idx: int):
    """Asynchronous facade around _solve_sync with a semaphore."""
    async with sema:
        return await asyncio.to_thread(_solve_sync, task_idx)

async def main():
    # kick off all jobs at once and wait for them to finish
    results = await asyncio.gather(*(solve_one(idx) for idx in tasks))

    # pretty‑print
    correct = 0
    for idx, ok, answer in results:
        print(f"Task {idx}: {'Correct' if ok else 'Incorrect'} - {answer}")
        correct += ok
    print(f"Correctly solved {correct} out of {samples} tasks.")

print(f'running with {args.solve_method} method')

asyncio.run(main())


