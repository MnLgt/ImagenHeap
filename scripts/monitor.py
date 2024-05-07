import time
import psutil
import GPUtil
from blessed import Terminal


def get_gpu_info():
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append(
            {
                "name": gpu.name,
                "load": int(gpu.load * 100),
                "memory_used": int(gpu.memoryUsed),
                "memory_total": int(gpu.memoryTotal),
                "memory_util": int(gpu.memoryUtil * 100),
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
