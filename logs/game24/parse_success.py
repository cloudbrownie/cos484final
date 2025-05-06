# open json
import os
import json

def parse_success(log_file):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist.")
        return

    with open(log_file, 'r') as f:
        logs = json.load(f)

    success_count = 0
    total_count = len(logs)

    for log in logs:
        infos = log.get('infos', [])
        if any(info.get('r', 0) == 1 for info in infos):
            success_count += 1
        #if infos[0].get('r', 0) == 1:
        #    success_count += 1
        #if sum(info.get('r', 0) for info in infos) >= 2:
        #    success_count += 1

    print(f"Total tasks: {total_count}, Successful tasks: {success_count}, Success rate: {success_count / total_count:.2%}")
    
if __name__ == "__main__":
    log_file = 'gpt-4.1-mini_0.7_astar_propose1_value3_greedy5_start901_end1000.json'  # Replace with your actual log file path
    parse_success(log_file)