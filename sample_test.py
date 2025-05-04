import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
import random

args = argparse.Namespace(backend='gpt-4.1-mini', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = Game24Task()

idx_min = 0
idx_max = 101
random_task_index = random.randint(idx_min, idx_max)
print(f"Random Task Index from {idx_min}-{idx_max}: {random_task_index}")

ys, infos = solve(args, task, random_task_index)
print(ys[0])
print(task.test_output(random_task_index, ys[0]))  # Print the output and its correctness

