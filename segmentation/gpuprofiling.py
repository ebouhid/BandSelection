import csv
import subprocess
import time

def track_gpu(interval, filename, stop_event):
    try:
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Time (s)', 'Power (W)', 'Memory Usage (MiB)'])
            start_time = time.time()
            while not stop_event.is_set():
                result_pow = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], universal_newlines=True)
                power_draw = float(result_pow.strip())
                result_mem = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], universal_newlines=True)
                memory_usage = int(result_mem.strip())
                current_time = time.time() - start_time
                csv_writer.writerow([current_time, power_draw, memory_usage])
                time.sleep(interval)
    except Exception as e:
        print(f"Error while tracking GPU: {e}")