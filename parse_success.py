# open json
import os
import json

def parse_success(log_file):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist.")
        return

    with open(log_file, 'r') as f:
        logs = json.load(f)

    one_shot_success = 0
    any_shot_success = 0
    total_successes = 0
    total_count = len(logs)

    for log in logs:
        infos = log.get('infos', [])
        if any(info.get('r', 0) == 1 for info in infos):
            any_shot_success += 1
        if infos[0].get('r', 0) == 1:
            one_shot_success += 1
        
        if len(infos) < 5:
            # add 0's to pad the list to 5
            infos += [{'r': 0}] * (5 - len(infos))
            
        #if sum(info.get('r', 0) for info in infos) >= 2:
        #    success_count += 1
    
    #acc_avg = sum(sum(r['infos'][j]['r'] for j in range(len(r['infos'])))
    #            / len(r['infos']) for r in logs) / len(logs)
    acc_avg = sum(sum(r['infos'][j]['r'] for j in range(len(r['infos'])))
                / len(r['infos']) for r in logs) / len(logs)
    acc_any = sum(any(info['r'] for info in r['infos']) for r in logs) / len(logs)
    one_shot_success = sum(1 for log in logs if log['infos'][0]['r'] == 1) / len(logs)
        

    #print(f"Total tasks: {total_count}, Successful tasks: {success_count}, Success rate: {success_count / total_count:.2%}")
    #print(f"Total tasks: {total_count}, One-shot success: {one_shot_success}, Any-shot success: {any_shot_success}")
    # print one shot and any shot success rate
    #print(f"Total tasks: {total_count}, One-shot success: {one_shot_success}, Any-shot success: {any_shot_success}")
    
    print(f"Total tasks: {total_count}")
    print(f"Any-shot accuracy: {acc_any:.2%}")
    print(f"Average accuracy: {acc_avg:.2%}")
    print(f"One-shot success rate: {one_shot_success:.2%}")
    #print(f"One-shot success rate: {one_shot_success / total_count:.2%}")
    #print(f"Any-shot success rate: {any_shot_success / total_count:.2%}")
    
if __name__ == "__main__":
    log_file = 'logs/game24/gpt-4.1-mini_0.7_astar_propose1_value3_greedy5_start900_end1000_oneshot2.json'  # Replace with your actual log file path
    parse_success(log_file)