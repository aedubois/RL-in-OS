import subprocess
import re

def convert_to_ms(value_str):
    """
    Converts a string representation of time to milliseconds.
    """
    if not value_str:
        return None
    
    value = float(value_str[:-2])  
    unit = value_str[-2:]  
    
    if unit == 'us':
        return value / 1000 
    elif unit == 'ms':
        return value
    elif unit == 's':
        return value * 1000
    return None

def run_wrk(url="http://localhost/server.html", duration=10, threads=2, connections=10):
    """
    Launches the wrk HTTP benchmarking tool and returns the requests per second and raw output.
    """
    cmd = [
        "wrk",
        "-t", str(threads),
        "-c", str(connections),
        "-d", f"{duration}s",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    match = re.search(r"Requests/sec:\s+([\d\.]+)", output)
    req_per_sec = float(match.group(1)) if match else None

    latency_match = re.search(r"Latency\s+(\d+\.\d+\w+)", output)
    p99_match = re.search(r"99%\s+(\d+\.\d+\w+)", output)
    latency = convert_to_ms(latency_match.group(1)) if latency_match else None
    p99 = convert_to_ms(p99_match.group(1)) if p99_match else None

    return req_per_sec, latency, p99, output

if __name__ == "__main__":
    rps, _, _, _ = run_wrk()