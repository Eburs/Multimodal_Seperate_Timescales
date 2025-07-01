import multiprocessing as mp
import subprocess
from typing import List, Tuple
import numpy as np
import torch as tc


class Task:
    def __init__(self, command, name):
        self.command = command
        self.name = name


class Argument:
    def __init__(self, name, values, add_to_name_as=None):
        self.name = name
        self.values = values
        self.add_to_name_as = add_to_name_as
        if len(values) > 1:
            print_statement = "please specify a name addition for argument {}, because it has more than one value".format(
                name
            )
            if name != "run":
                assert add_to_name_as is not None, print_statement


def get_current_gpu_utilization():
    """
    From: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are GPU utilization in %.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_util = [int(x) for x in result.strip().split("\n")]

    # same for memory usage
    mem_used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # and max memory
    mem_max = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    gpu_mem = [
        int(100 * x / y)
        for x, y in zip(
            [int(x) for x in mem_used.strip().split("\n")],
            [int(x) for x in mem_max.strip().split("\n")],
        )
    ]
    return dict(zip(range(len(gpu_util)), gpu_util)), dict(
        zip(range(len(gpu_mem)), gpu_mem)
    )


def add_argument(tasks, arg):
    new_tasks = []
    for task in tasks:
        for arg_value in arg.values:
            if type(arg_value) is list:
                arg_name = "-".join([str(i) for i in arg_value])
                new_name = add_to_name(task, arg, arg_name)
                arg_command = " ".join([str(i) for i in arg_value])
                new_command = " ".join(
                    [task.command, "--{}".format(arg.name), arg_command]
                )
            else:
                new_name = add_to_name(task, arg, arg_value)
                new_command = " ".join(
                    [task.command, "--{}".format(arg.name), str(arg_value)]
                )
            new_tasks.append(Task(new_command, new_name))
    return new_tasks


def add_to_name(task, arg, arg_value):
    if arg.add_to_name_as is not None:
        new_name = "".join([task.name, arg.add_to_name_as, str(arg_value).zfill(2)])
    else:
        new_name = task.name
    return new_name


def check_arguments_for_gpu(args: List[Argument]) -> bool:
    """
    Check the ubermain arguments for the 'use_gpu' flag
    and general cuda availability.
    """
    use_gpu = False
    # if the user specifies device ids himself,
    # do not bother distributing the tasks.
    for arg in args:
        if arg.name == "device_id":
            print("Device id(s) specified by user -> manual task distribution")
            return False
        elif arg.name == "use_gpu":
            if not 0 in arg.values:
                assert tc.cuda.is_available(), "CUDA is not available."
                print("'use_gpu' flag is set.")
                use_gpu = True
    if use_gpu and tc.cuda.is_available():
        print("Will distribute tasks to GPUs automatically.")
    return use_gpu


def distribute_tasks_across_gpus(
    tasks: List[Task], n_proc_per_gpu: int, n_cpu: int
) -> Tuple[List, int]:
    """
    Checks current GPU utilization of the machine,
    picks out idle devices and distributes them
    across tasks.
    """
    util_dict, mem_dict = get_current_gpu_utilization()
    # filter device ids of unused GPUs
    device_ids = []
    for id_, util in util_dict.items():  # TODO: clean up
        # TODO: maybe change the criterion to softer constraint ...
        # Adjusted to allow more than 5 mb
        # utelisation because my opperating system
        # sometimes uses more
        print(f"gpu utilisation: {util}, memory usage: {mem_dict[id_]}")
        if util < 50.0 and mem_dict[id_] < 1000.0:
            device_ids.append(id_)

    # are all GPUs in use?
    # TODO: maybe a bit harsh to throw RuntimeError here ...
    if not device_ids:
        raise RuntimeError("All GPUs of the machine are in use!")

    # check if there are too many parallel processes spawned by user
    # compared to available GPUs
    device_distribution = np.repeat(device_ids, min(n_cpu, n_proc_per_gpu))
    sz = device_distribution.size
    if sz < n_cpu:
        print(
            "There are not enough GPU Resources available to spawn "
            f"{n_cpu} processes. Reducing number of parallel runs "
            f"to {sz}"
        )
        new_n_cpu = sz
    else:
        new_n_cpu = n_cpu

    # distribute devices across tasks
    new_tasks = []
    idx = 0
    for task in tasks:
        arg = Argument("device_id", [device_distribution[idx]])
        new_tasks.append(*add_argument([task], arg))
        idx += 1
        if idx == new_n_cpu:
            idx = 0
    return new_tasks, new_n_cpu


def create_tasks_from_arguments(
    args: List[Argument], n_proc_per_gpu: int, n_cpu: int
) -> Tuple[List, int]:
    # check args for gpu usage
    use_gpu = check_arguments_for_gpu(args)

    tasks = [Task(command="python3 main.py", name="")]
    for arg in args:
        tasks = add_argument(tasks, arg)

    pp = n_cpu
    if use_gpu:
        # tasks w/ device id, number of parallel processes
        tasks, pp = distribute_tasks_across_gpus(tasks, n_proc_per_gpu, n_cpu)

    for task in tasks:
        task.command = " ".join([task.command, "--name", task.name])

    return tasks, pp


def run_settings(tasks, n_cpu):
    pool = mp.Pool(processes=n_cpu)
    pool.map(process_task, tasks, chunksize=1)
    pool.close()
    pool.join()


def process_task(task):
    subprocess.call(task.command.split())
