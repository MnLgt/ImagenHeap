import time
import psutil
import GPUtil
from blessed import Terminal


import math  # Import math to check for NaN


def get_gpu_info():
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        load = gpu.load if gpu.load is not None else 0
        memory_used = gpu.memoryUsed if gpu.memoryUsed is not None else 0
        memory_total = gpu.memoryTotal if gpu.memoryTotal is not None else 0
        memory_util = gpu.memoryUtil if gpu.memoryUtil is not None else 0

        # Replace NaN values with 0
        load = 0 if math.isnan(load) else int(load * 100)
        memory_util = 0 if math.isnan(memory_util) else int(memory_util * 100)

        gpu_info.append(
            {
                "name": gpu.name,
                "load": load,
                "memory_used": int(memory_used),
                "memory_total": int(memory_total),
                "memory_util": memory_util,
            }
        )
    return gpu_info


def get_cpu_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    return cpu_percent


def get_memory_info():
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "used": memory.used,
        "percent": memory.percent,
    }


def draw_progress_bar(used, total, bar_length=20):
    percent = (used / total) * 100
    bar_filled_length = int(round(bar_length * used / total))
    bar = "█" * bar_filled_length + "-" * (bar_length - bar_filled_length)
    return f"|{bar}| {percent:.2f}%"


def main():
    term = Terminal()
    with term.fullscreen(), term.hidden_cursor():
        while True:
            term.clear()
            cpu_info = get_cpu_info()
            memory_info = get_memory_info()
            gpu_info = get_gpu_info()

            with term.location(0, 0):
                print(f"CPU Usage: {draw_progress_bar(cpu_info, 100)}")
                print(
                    f"Memory Usage: {draw_progress_bar(memory_info['used'], memory_info['total'])}"
                )

                for idx, gpu in enumerate(gpu_info):
                    print(
                        f"GPU-{idx} {gpu['name']} Load: {draw_progress_bar(gpu['load'], 100)}"
                    )
                    print(
                        f"GPU-{idx} {gpu['name']} Memory: {draw_progress_bar(gpu['memory_used'], gpu['memory_total'])}"
                    )

            time.sleep(1)


if __name__ == "__main__":
    main()
