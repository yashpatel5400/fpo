# VERY hacky script but hey, gets the job done
import argparse
import os
import libtmux
import time

import utils

def main(pde, cutoff):
    results_dir = utils.RESULTS_DIR(pde)
    fs, _ = utils.get_data(args.pde, train=False)
    
    cur_sample_idx = 0
    batch_size = 10
    dispatched = False
    server = None
    while cur_sample_idx < len(fs):
        if dispatched:
            ref_result_fn = os.path.join(results_dir, f"{cur_sample_idx}.csv")
            print(f"Checking {ref_result_fn}...")
            if not os.path.exists(ref_result_fn):
                time.sleep(20)
            else:
                cur_sample_idx += batch_size
                dispatched = False
        else:
            server = libtmux.Server()
            while len(server.sessions) > 0:
                server.kill_session(f'${len(server.sessions)-1}')
            for sample_idx in range(cur_sample_idx, cur_sample_idx + batch_size):
                result_fn = os.path.join(results_dir, f"{sample_idx}.csv")
                if os.path.exists(result_fn):
                    continue

                server.new_session(attach=False)
                session = server.get_by_id(f'${len(server.sessions)-1}')
                p = session.attached_pane
                p.send_keys("conda activate spectral", enter=True)
                cmd = f"python fpo.py --pde {pde} --sample {sample_idx} --cutoff {cutoff}"
                p.send_keys(cmd, enter=True)
                print(f"{cmd}")
                dispatched = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    parser.add_argument("--cutoff", type=int, default=8)
    args = parser.parse_args()
    main(args.pde, args.cutoff)