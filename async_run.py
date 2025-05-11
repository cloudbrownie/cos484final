import asyncio
import os, json, argparse, time
from tot.tasks import get_task
from tot.methods.bfs   import solve, naive_solve
from tot.methods.astar import solve_astar
from tot.methods.mcts  import solve_mcts
from tot.models import gpt_usage

# ---------- NEW: async driver ---------- #
async def run_async(args):
    task   = get_task(args.task)

    # pick solver & logfile name exactly as before -------------
    if args.search_algo == 'bfs' and args.naive_run:
        solver = naive_solve
        file   = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    elif args.search_algo == 'bfs':
        solver = solve
        file   = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    elif args.search_algo == 'astar':
        solver = solve_astar
        file   = f'./logs/{args.task}/{args.backend}_{args.temperature}_astar_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_oneshot2.json'
    elif args.search_algo == 'mcts':
        solver = solve_mcts
        file   = f'./logs/{args.task}/{args.backend}_{args.temperature}_mcts_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        raise ValueError("Invalid search algorithm")

    os.makedirs(os.path.dirname(file), exist_ok=True)

    # --- helpers -----------------------------------------------------------
    async def run_one(idx:int):
        """
        Executes ONE task in a thread so we don't have to touch the (sync) solver
        while still letting the event-loop schedule many in parallel.
        """
        ys, info   = await asyncio.to_thread(solver, args, task, idx)

        infos      = [task.test_output(idx, y) for y in ys]
        info.update({'idx': idx,
                     'ys': ys,
                     'infos': infos,
                     'usage_so_far': gpt_usage(args.backend)})
        return info

    # -------- launch -------------------------------------------------------
    start, end  = args.task_start_index, args.task_end_index
    
    if args.task == 'game24':
        indices = list(range(start, end))
    elif args.task == 'text':
        indices = list(range(start, end))
    elif args.task == 'crosswords':
        indices = list(range(start, end))
    
    
    start_time = time.time()
    print(f"⇢  Running indices [{start}, {end}) with up to {args.concurrency} workers "
          f"at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    semaphore   = asyncio.Semaphore(args.concurrency)

    async def bounded(idx):
        async with semaphore:
            return await run_one(idx)

    #results     = await asyncio.gather(*(bounded(i) for i in range(start, end)))
    results     = await asyncio.gather(*(bounded(i) for i in indices))

    # -------- bookkeeping --------------------------------------------------
    results.sort(key=lambda r: r["idx"])           # keep log order deterministic
    with open(file, "w") as f:
        json.dump(results, f, indent=4)

    acc_avg = sum(sum(r['infos'][j]['r'] for j in range(len(r['infos'])))
                  / len(r['infos']) for r in results) / len(results)
    acc_any = sum(any(info['r'] for info in r['infos']) for r in results) / len(results)

    print(f"✓ finished [{start}, {end})  avg-acc={acc_avg:.3f}  any-acc={acc_any:.3f}")
    print("usage_so_far", gpt_usage(args.backend))
    print(f'spent {time.time() - start_time:.2f}s')
# ------------------------------------------------------------------------- #

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'o1-mini', 'gpt-4.1-mini'], default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--search_algo', type=str, choices=['bfs', 'astar', 'mcts'], default='bfs')
    
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    # add one small flag:
    args.add_argument('--concurrency', type=int, default=4,
                   help='how many task indices to run simultaneously')
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_async(args))
