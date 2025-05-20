import subprocess
import re

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
    if match:
        req_per_sec = float(match.group(1))
    else:
        req_per_sec = None

    return req_per_sec, output

if __name__ == "__main__":
    rps, _ = run_wrk()