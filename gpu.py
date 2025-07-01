import multitasking

util_dict, mem_dict = multitasking.get_current_gpu_utilization()
for gpu_id, util in util_dict.items():
    print(f"GPU {gpu_id}: {util}% utilization, {mem_dict[gpu_id]} MB memory used")
