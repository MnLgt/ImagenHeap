import psutil
import time


def get_cpu_memory():
    # Retrieve memory stats
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024**3)  # Convert bytes to GB
    used_memory = memory.used / (1024**3)  # Convert bytes to GB
    return total_memory, used_memory


def print_memory_bar(total_memory, used_memory):
    # Calculate the percentage of used memory
    used_percent = (used_memory / total_memory) * 100
    # Calculate the length of the bar
    bar_length = 50
    filled_length = int(round(bar_length * used_memory / total_memory))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    # Print the bar with the percentage
    print(f"CPU Memory Usage: |{bar}| {used_percent:.2f}% used")


def main():
    try:
        while True:
            total_memory, used_memory = get_cpu_memory()
            print_memory_bar(total_memory, used_memory)
            time.sleep(1)  # Refresh every 1 second
    except KeyboardInterrupt:
        print("Monitoring stopped.")


if __name__ == "__main__":
    main()
