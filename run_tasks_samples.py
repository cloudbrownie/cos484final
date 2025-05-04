import argparse
from tot.methods.bfs import solve
from tot.methods.astar import solve_astar
from tot.methods.mcts import solve_mcts
from tot.tasks.game24 import Game24Task
from tot.models import gpt_usage
import random
import asyncio
from functools import partial
import time

args = argparse.ArgumentParser()
args.add_argument('--backend', type=str, default='gpt-4.1-mini')
args.add_argument('--temperature', type=float, default=0.7)
args.add_argument('--task', type=str, default='game24')
args.add_argument('--naive_run', type=bool, default=False)
args.add_argument('--prompt_sample', type=str, default=None)
args.add_argument('--method_generate', type=str, default='propose')
args.add_argument('--method_evaluate', type=str, default='value')
args.add_argument('--method_select', type=str, default='greedy')
args.add_argument('--n_generate_sample', type=int, default=1)
args.add_argument('--n_evaluate_sample', type=int, default=3)
args.add_argument('--n_select_sample', type=int, default=5)
args.add_argument('--solve_method', type=str, default='bfs', choices=['bfs', 'astar', 'mcts'], help='Method to solve the task')  
args = args.parse_args()

print(args)

task = Game24Task()

idx_min = 901
idx_max = 1000

## sample random numbers from range without replacement
#samples = 50
#tasks = random.sample(range(idx_min, idx_max + 1), samples)
tasks = list(range(idx_min, idx_max + 1))

# limit concurrent threads so you stay under OpenAI’s parallel‑request quota
CONCURRENCY = 10
sema = asyncio.Semaphore(CONCURRENCY)


if args.solve_method == 'bfs':
    solve_func = solve
elif args.solve_method == 'astar':
    solve_func = partial(solve_astar, beam_width=args.n_select_sample)
elif args.solve_method == 'mcts':
    solve_func = partial(solve_mcts, n_simulations=20)

# ---- thin wrappers --------------------------------------------------------
def _solve_sync(task_idx: int):
    """Run the original blocking solve() in a worker thread."""
    print(f"Solving task {task_idx}: {task.get_input(task_idx)}")
    #ys, _ = solve(args, task, task_idx, to_print=False)
    ys, _ = solve_func(args, task, task_idx, to_print=False)
    
    if type(ys) == list:
        ys = ys[0]
    
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
    print(f"Correctly solved {correct} out of {len(tasks)} tasks.")

print(f'running with {args.solve_method} method')
print(f'start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

asyncio.run(main())

print(f'{args.solve_method} method finished\nCosted {gpt_usage(args.backend)}')
print(f'end time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
