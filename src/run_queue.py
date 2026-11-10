"""Run a list of GPU jobs, one job per GPU at a time, skipping jobs whose output already exists.

Usage: python run_queue.py jobs/<name>.json [--gpus 0 1 2 3]

The job file is a JSON list of {"name", "cmd", "done"}: `cmd` is a shell command run from this directory with
CUDA_VISIBLE_DEVICES set, `done` a path whose existence means the job has finished. Each job's output goes to
logs/<name>.log; a summary line per job is printed when it ends. Safe to re-run after an interruption.
With --lock, a job is claimed through logs/<name>.lock (holding the queue's PID), so several queues can share one
job file (e.g. one on GPU 2 now, another on GPUs 0 1 later); locks of dead queues are ignored.
"""
import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def lock_path(job):
    return os.path.join(HERE, "logs", job["name"] + ".lock")


def claim(job):
    """Create the job's lock file atomically; a lock whose queue process is dead is taken over."""
    path = lock_path(job)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                os.kill(int(open(path).read().strip() or 0), 0)
                return False  # another live queue runs it
            except (ProcessLookupError, ValueError):
                os.remove(path)  # stale lock
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--lock", action="store_true", help="claim jobs with lock files (several queues, one job file)")
    args = ap.parse_args()
    pending = [j for j in json.load(open(args.jobs)) if not os.path.exists(os.path.join(HERE, j["done"]))]
    print(f"{len(pending)} jobs to run", flush=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    env = dict(os.environ, HF_HOME=os.environ.get("HF_HOME", os.path.join(os.path.dirname(HERE), ".hf_cache")))
    running = {}  # gpu -> (job, process, start)
    failed = 0
    while pending or running:
        for gpu in args.gpus:
            while gpu not in running and pending:
                job = pending.pop(0)
                if os.path.exists(os.path.join(HERE, job["done"])) or (args.lock and not claim(job)):
                    continue
                log = open(os.path.join(HERE, "logs", job["name"] + ".log"), "w")
                p = subprocess.Popen(job["cmd"], shell=True, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(env, CUDA_VISIBLE_DEVICES=gpu))
                running[gpu] = (job, p, time.time())
                print(f"start {job['name']} on GPU {gpu}", flush=True)
                break
        for gpu, (job, p, t0) in list(running.items()):
            if p.poll() is not None:
                ok = p.returncode == 0 and os.path.exists(os.path.join(HERE, job["done"]))
                failed += not ok
                print(f"{'done' if ok else 'FAILED'} {job['name']} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                del running[gpu]
                if args.lock and os.path.exists(lock_path(job)):
                    os.remove(lock_path(job))
        time.sleep(5)
    print(f"all finished, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
