import re
import subprocess
import time


def get_gpu_memory():
    # Run the nvidia-smi command to get GPU details
    smi_output = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,nounits,noheader",
            ]
        )
        .decode()
        .strip()
    )
    # Parse the total and used memory
    total_memory, used_memory = map(int, re.split(",\s*", smi_output))
    return total_memory, used_memory


def print_memory_bar(total_memory, used_memory):
    # Calculate the percentage of used memory
    used_percent = (used_memory / total_memory) * 100
    # Calculate the length of the bar
    bar_length = 50
    filled_length = int(round(bar_length * used_memory / total_memory))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    # Print the bar with the percentage
    print(f"GPU Memory Usage: |{bar}| {used_percent:.2f}% used")


def main():
    try:
        while True:
            total_memory, used_memory = get_gpu_memory()
            print_memory_bar(total_memory, used_memory)
            time.sleep(1)  # Refresh every 1 second
    except KeyboardInterrupt:
        print("Monitoring stopped.")


if __name__ == "__main__":
    main()
