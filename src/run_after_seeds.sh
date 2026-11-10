#!/bin/sh
# Wait for the v0 seeds queue, then run the Table 6 GPU jobs (MovieLens-H train+eval, Amazon-H probability eval).
cd "$(dirname "$0")"
while pgrep -f "run_queue.py jobs/v0_seeds_mistral.json" > /dev/null; do sleep 60; done
echo "seeds queue finished $(date)"
python run_queue.py jobs/table6_mistral.json --gpus ${GPUS:-0 1 2}
echo "TABLE6 GPU JOBS FINISHED $(date)"
