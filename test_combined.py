from multitasking import *
import torch
from pathlib import Path


def prepare_data_paths(subsystem_dims, experiment):
    """
    Specify the paths to the training and evaluation data.
    Give the dimensions of the subsystems as an array
    """
    train_data_path = (
        "./data/correlated_system/lorenz_bursting_1vec_correlated_train_reshaped.pt"
    )
    test_data_path = (
        "./data/correlated_system/lorenz_bursting_1vec_correlated_test_reshaped.pt"
    )
    train_data = torch.load(train_data_path)
    test_data = torch.load(test_data_path)

    number_subsystems = len(subsystem_dims)
    current_dim = 0
    new_train_data_paths = []
    new_test_data_paths = []
    for sys_number, dim in enumerate(subsystem_dims):
        # Partition the data into subsystems
        system_train_data = train_data[sys_number::number_subsystems].clone()
        system_test_data = test_data[sys_number::number_subsystems].clone()

        # write the data to new files
        system_train_data_path = (
            Path(train_data_path).parent / f"{experiment}_{sys_number}train.pt"
        )
        system_test_data_path = (
            Path(test_data_path).parent / f"{experiment}_{sys_number}test.pt"
        )
        torch.save(system_train_data, system_train_data_path)
        torch.save(system_test_data, system_test_data_path)
        new_train_data_paths.append(system_train_data_path)
        new_test_data_paths.append(system_test_data_path)

        current_dim += dim

    return train_data_path, test_data_path, new_train_data_paths, new_test_data_paths


def ubermain(
    train_data_path,
    eval_data_path,
    system_dimension,
    total_dimension,
    experiment,
    shared_objects=1,
):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6.
    Possible Arguments can be found in main.py

    When using GPU for training (i.e. Argument 'use_gpu 1')  it is generally
    not necessary to specify device ids, tasks will be distributed automatically.

    To account for the different subsystems, their dimension and paths need to be
    passed as arguments. Also shared_objects for the full system. For a fair comparison, also pass the total dimensions.

    Specification of the experiment name is also moved to the arguments,
    for naming purposes.
    """
    args = []

    args.append(Argument("num_epochs", [1000]))
    args.append(Argument("batch_size", [1024]))
    args.append(Argument("batches_per_epoch", [50]))

    args.append(
        Argument(
            "data_path",
            [train_data_path],
        )
    )
    args.append(
        Argument(
            "eval_data_path",
            [eval_data_path],
        )
    )
    args.append(Argument("train_set_size", [90000]))

    args.append(Argument("experiment", [experiment]))
    args.append(Argument("run", list(range(1, 1 + n_runs))))

    args.append(Argument("hierarchisation_scheme", ["projection"], add_to_name_as=""))
    args.append(
        Argument("num_individual_params", [system_dimension], add_to_name_as="dp")
    )

    args.append(Argument("obs_model", ["identity"]))
    args.append(Argument("obs_size", [system_dimension]))
    args.append(Argument("latent_size", [40 // total_dimension * system_dimension]))
    args.append(Argument("forcing_size", [system_dimension]))
    args.append(
        Argument(
            "hidden_size",
            [50 // total_dimension * system_dimension],
            add_to_name_as="dh",
        )
    )

    args.append(Argument("learning_rate", [1e-4]))
    args.append(Argument("individual_learning_rate", [1e-3]))

    args.append(Argument("tf_alpha_start", [0.2]))
    args.append(Argument("tf_alpha_end", [0.02]))

    args.append(Argument("seq_len", [40]))

    args.append(Argument("weight_decay", [0]))

    # args.append(Argument("compile", [""]))
    args.append(Argument("use_gpu", [""]))

    args.append(Argument("metrics", ["kl pse"]))
    args.append(Argument("plots", ["pow hier 3D"]))

    args.append(Argument("lam", [0]))

    # args.append(Argument('clip_grad_norm', [10]))

    args.append(Argument("learn_noise_cov", [""]))

    args.append(Argument("num_workers", [4]))

    args.append(Argument("num_shared_objects", [shared_objects]))

    return args


if __name__ == "__main__":
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # number of runs for each experiment
    n_runs = 1
    # number of processes run parallel on a single GPU
    n_proc_per_gpu = 1
    # number of runs to run in parallel
    n_cpu = 1 * n_proc_per_gpu

    # Specify the dimensions of the subsystems and the experiment name
    subsystem_dims = [3, 3]
    total_dim = sum(subsystem_dims)
    experiment_name = "lorenzbursting"

    # get the paths to the data
    (
        full_system_data_path,
        full_system_eval_data_path,
        subsystem_train_paths,
        subsystem_eval_paths,
    ) = prepare_data_paths(subsystem_dims, experiment_name)

    # prepare the tasks for the full system
    full_system_args = ubermain(
        train_data_path=full_system_data_path,
        eval_data_path=full_system_eval_data_path,
        system_dimension=total_dim,
        total_dimension=total_dim,
        experiment=experiment_name,
        shared_objects=len(subsystem_dims),
    )
    tasks, pp = create_tasks_from_arguments(full_system_args, n_proc_per_gpu, n_cpu)

    # add the subsystem-tasks
    for sys_number, dim in enumerate(subsystem_dims):
        args = ubermain(
            train_data_path=subsystem_train_paths[sys_number],
            eval_data_path=subsystem_eval_paths[sys_number],
            system_dimension=dim,
            total_dimension=total_dim,
            experiment=f"{experiment_name}_{sys_number}",
        )
        tasks_sys, _ = create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu)
        tasks += tasks_sys

    run_settings(tasks, pp)
