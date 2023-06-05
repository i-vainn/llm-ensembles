import sys
import subprocess
import time
import os


def extract_results_path(command):
    for arg in command:
        if arg.startswith("results_path="):
            return arg.split("=")[1]
    return None

if __name__ == "__main__":
    # Extract results_path from the command
    results_path = extract_results_path(sys.argv)

    # Set up the environment for the subprocess
    env = os.environ.copy()

    # Record start time and run the experiment command
    start_time = time.time()

    process = subprocess.Popen(sys.argv[1:], capture_output=True, text=True, env=env)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save the stats to a file
    if results_path:
        stats_file = os.path.join(results_path, "stats.txt")
        with open(stats_file, "w") as f:
            f.write(f"Peak CUDA memory usage: {peak_memory / (1024 ** 2)} MB\n")
            f.write(f"Execution time: {elapsed_time:.2f} seconds\n")

    if process.returncode == 0:
        print("The experiment finished successfully.")
        print("Output:", process.stdout.strip())
    else:
        print("The experiment finished with an error.")
        print("Error:", process.stderr.strip())

    # stdout, stderr = process.communicate()
    # if process.returncode == 0:
    #     print("The experiment finished successfully.")
    #     # print("Output:", stdout.strip())
    # else:
    #     print("The experiment finished with an error.")
    # #     print("Error:", stderr.strip())
