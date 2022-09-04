import glob
import os
import json
import argparse
from scipy.stats import bootstrap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of the evaluation, this is used as the output directory name.")
    parser.add_argument("-p", "--participant", type=str, default=None, help="Show result for only this participant.")
    parser.add_argument("-t", "--notes", dest='notes', action='store_true', help="If set, olnly participant notes are printed.")
    parser.add_argument("-r", "--root", type=str, default="outputs", help="Root directory.")
    parser.set_defaults(notes=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    results = {}
    for f_name in glob.glob(os.path.join(args.root, args.name) + "/*.json"):
        if args.participant is not None:
            if f_name != args.participant:
                continue
        with open(f_name, 'r') as f:
            results[f_name] = json.load(f)

    assert len(results) > 0, f"Study: {args.name} has no result to be processed!"

    systems = None
    for participant_id in results:
        result = results[participant_id]
        if systems is None:
            systems = set(result['system_names'])
        assert systems == set(result['system_names']), f"The study folder contains an inconsistent result file: {participant_id}!"
    systems = sorted(list(systems))

    total_conversations = 0
    total_turns = 0

    for result in results.values():
        result['valid_checks'] = 0
        result['total_checks'] = 0

        for conv_id, conv in result['sanity_checks'].items():
            total_conversations += 1
            for idx, sys_id in enumerate(conv):
                for turn_idx, turn_check in enumerate(conv[sys_id]):
                    if idx == 0:
                        total_turns += 1
                    
                    if not turn_check:
                        continue
                    result['total_checks'] += 1
                    check_rating = result['ratings'][conv_id][sys_id][turn_idx]
                    for another_sys_id in result['ratings'][conv_id]:
                        if sys_id == another_sys_id:
                            continue
                        another_rating = result['ratings'][conv_id][another_sys_id][turn_idx]
                        #if another_rating > check_rating:
                        if another_rating >= check_rating:
                            break
                    else:
                        result['valid_checks'] += 1

    print("")

    if args.notes:
        print(f"========== Showing notes left by participants of {args.name} ==========")
        for id in results:
            print(f"{id}(valid {results[id]['valid_checks']}/{results[id]['total_checks']}): \t {results[id]['note']}")

    else:

        import numpy as np
        import scikit_posthocs as sp
        import scipy.stats as ss
        
        print(f"========== Showing results of the study {args.name} ==========")
        print("")

        print(f"Number of conversations: {total_conversations}")
        print(f"Number of turns: {total_turns}")
        print("")

        valid_users = sum(r['valid_checks'] == r['total_checks'] for r in results.values())
        all_users = len(results)
        print(f"Participants: {valid_users}/{all_users} (valid/total)")

        valid_checks = sum(r['valid_checks'] for r in results.values())
        all_checks = sum(r['total_checks'] for r in results.values())
        print(f"Sanity checks: {valid_checks}/{all_checks} (valid/total)")
        print("")
        
        if valid_checks == 0 and valid_checks != all_checks:
            print("There are no valid trials! Skipping calculation of statistics.")
        else:
            data = []
            for sys_id in systems:
                sys_data = []
                for result in results.values():
                    if result['valid_checks'] != result['total_checks']:
                        continue
                    for conv_id in result['dialog_ids']:
                        sanity_mask = [False for _ in result['ratings'][conv_id][sys_id]]
                        for s in result['sanity_checks'][conv_id]:
                            sanity_mask = sanity_mask or result['sanity_checks'][conv_id][s]
                        wo_sanity_checks = [x for x, y in zip(result['ratings'][conv_id][sys_id], sanity_mask) if not y]
                        sys_data.extend(wo_sanity_checks)
                data.append(sys_data)

            data = np.array(data, dtype=object)
            data = data.astype(float)

            for j in range(len(data)):
                print(j)
                for i in range(4):
                    print(f"Ranked {i+1}: {100 * np.mean(data[j] == i+1)}")

            print(f"Mean rankings: {np.round(np.mean(data, axis=1), decimals=2)}")
            print(f"Std rankings: {np.round(np.std(data, axis=1), decimals=2)}")

            rng = np.random.default_rng()
            print(len(data))
            for i in range(len(data)):
                res = bootstrap((data[i],), np.std, axis=-1, confidence_level=0.95, n_resamples=1000, random_state=rng)
                print(res)

            

            friedman_results = ss.friedmanchisquare(*data)
            print(f"All systems have the same performance with probability {friedman_results.pvalue}!")

            nemenyi_results = sp.posthoc_nemenyi_friedman(data.T) 
            print("Pairwise probability of the same performance:")
            print(nemenyi_results)

    print("")
