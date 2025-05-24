import time
import subprocess
from datetime import datetime
import os

def simulate_sensor_log():
    """
    Simulates a sensor reading (e.g., temperature) and writes it to a log file.
    """
    timestamp = datetime.now().isoformat()
    sensor_value = round(20 + 5 * (os.getpid() % 10), 2)  # Pseudo-random value
    with open("Third Scenario - IoT/iot_sensor.log", "a") as f:
        f.write(f"{timestamp},sensor={sensor_value}°C\n")
    print(f"[SENSOR] {timestamp} → {sensor_value}°C")

def simulate_light_computation():
    """
    Simulates a lightweight task such as quick file compression.
    """
    with open("dummy.txt", "w") as f:
        f.write("data" * 1000)
    subprocess.run("zip -q dummy.zip dummy.txt", shell=True)
    os.remove("dummy.txt")
    os.remove("dummy.zip")
    print("[TASK] compression simulated")

def simulate_disk_access():
    """
    Simulates disk I/O activity (simple log write).
    """
    with open("Third Scenario - IoT/disk_write.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} – Disk write test\n")
    print("[DISK] simulated disk access")

def simulate_periodic_task():
    """
    Executes a full round of simulated IoT tasks.
    """
    simulate_sensor_log()
    simulate_light_computation()
    simulate_disk_access()

if __name__ == "__main__":
    print("IoT periodic simulation started (CTRL+C to stop)...\n")
    try:
        while True:
            simulate_periodic_task()
            time.sleep(30)  # Wait 30 seconds between each task
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
