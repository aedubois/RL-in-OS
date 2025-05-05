#!/bin/bash
# Simule une charge système active pour tester l'agent

echo "Simulation de charge système en cours..."

# Nombre de cœurs pour simuler la charge (ajustez selon votre machine)
NUM_CORES=$(nproc)

while true; do
  # Simuler une charge CPU (50-80% sur 50% des cœurs)
  stress --cpu $((NUM_CORES / 2)) --timeout 10s &

  # Simuler une utilisation RAM (512 Mo sur 4 workers)
  stress --vm 4 --vm-bytes 512M --timeout 10s &

  # Simuler des bursts de Swap (rare mais significatif)
  dd if=/dev/zero of=/tmp/swap_test bs=1M count=256 oflag=dsync 2>/dev/null
  rm -f /tmp/swap_test &

  # Simuler des I/O intenses (création et suppression de fichiers)
  for i in {1..50}; do
    dd if=/dev/zero of=tempfile_$i bs=1M count=1 oflag=dsync 2>/dev/null
    rm -f tempfile_$i
  done &

  # Simuler une latence réseau (ajoutez un délai artificiel)
  ping -c 5 -i 0.2 8.8.8.8 >/dev/null 2>&1 &

  # Pause entre les cycles pour simuler un système actif
  sleep 5
done