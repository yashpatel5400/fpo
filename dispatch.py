# VERY hacky script but hey, gets the job done
import argparse
import os
import libtmux

import utils

def main(results_dir):
    server = libtmux.Server()

    for sample_idx in range(0,25):
        result_fn = os.path.join(results_dir, f"{sample_idx}.csv")
        if os.path.exists(result_fn):
            continue

        server.new_session(attach=False)
        session = server.get_by_id(f'${len(server.sessions)-1}')
        p = session.attached_pane
        p.send_keys("conda activate spectral", enter=True)
        cmd = f"python fpo.py --pde navier_stokes --sample {sample_idx}"
        p.send_keys(cmd, enter=True)
        print(f"{cmd}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    args = parser.parse_args()
    main(utils.RESULTS_DIR(args.pde))