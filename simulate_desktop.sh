#!/bin/bash
# Simulates an active system load to test the agent

echo "Simulating system load..."

# Number of cores to simulate the load
NUM_CORES=$(nproc)

while true; do
  # Simulate CPU load (50-80% on 50% of the cores)
  stress --cpu $((NUM_CORES / 2)) --timeout 10s &

  # Simulate RAM usage (512 MB on 4 workers)
  stress --vm 4 --vm-bytes 512M --timeout 10s &

  # Simulate Swap bursts (rare but significant)
  dd if=/dev/zero of=/tmp/swap_test bs=1M count=256 oflag=dsync 2>/dev/null
  rm -f /tmp/swap_test &

  # Simulate intense I/O (creating and deleting files)
  for i in {1..50}; do
    dd if=/dev/zero of=tempfile_$i bs=1M count=1 oflag=dsync 2>/dev/null
    rm -f tempfile_$i
  done &

  # Simulate network latency (add artificial delay)
  ping -c 5 -i 0.2 8.8.8.8 >/dev/null 2>&1 &

  # Pause between cycles to simulate an active system
  sleep 5
done