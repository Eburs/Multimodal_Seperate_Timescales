import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm


def roessler(t, x, a, b, c):
    dxdt = -x[1] - x[2]
    dydt = x[0] + a * x[1]
    dzdt = b + x[2] * (x[0] - c)
    return [dxdt, dydt, dzdt]


def mInf(V, VhNa, kNa):
    mInf = 1 / (1 + np.exp((VhNa - V) / kNa))
    return mInf


def hInf(V, VhM, kM):
    hInf = 1 / (1 + np.exp((VhM - V) / kM))
    return hInf


def nInf(V, Vhk, kk):
    nInf = 1 / (1 + np.exp((Vhk - V) / kk))
    return nInf


def sigma(V):
    return 1 / (1 + 0.33 * np.exp(-V * 0.0625))


def bursting_neuron(
    t,
    x,
    Cm=6,
    gl=8,
    El=-80,
    gNa=20,
    ENa=60,
    VhNa=-20,
    kNa=15,
    gk=10,
    Ek=-90,
    Vhk=-25,
    kk=5,
    tauN=1,
    gM=25,
    VhM=-15,
    kM=5,
    tauH=200,
    gNMDA=10.2,
):
    V = x[0]
    n = x[1]
    h = x[2]
    dVdt = (
        -gl * (V - El)
        - gNa * mInf(V, VhNa, kNa) * (V - ENa)
        - gk * n * (V - Ek)
        - gM * h * (V - Ek)
        - gNMDA * sigma(V) * V
    ) / Cm
    dndt = (nInf(V, Vhk, kk) - n) / tauN
    dhdt = (hInf(V, VhM, kM) - h) / tauH
    return [dVdt, dndt, dhdt]


def lorenz(t, x, sigma, beta, rho):
    dxdt = sigma * (x[1] - x[0])
    dydt = x[0] * (rho - x[2]) - x[1]
    dzdt = x[0] * x[1] - beta * x[2]
    return [dxdt, dydt, dzdt]


def lorenz_bursting_correlated():
    """
    Generates correlated data from the Lorenz and bursting neuron systems.
    Saves the training and test datasets as PyTorch tensors.
    """
    subjects = 50
    tsteps = 100000
    training_data = torch.zeros((subjects * 2, tsteps, 3))
    test_data = torch.zeros((subjects * 2, tsteps, 3))
    t_lorenz = 1000
    t_bursting = 50000
    bursting_timesteps = np.linspace(0, t_bursting, tsteps)
    lorenz_timesteps = np.linspace(0, t_lorenz, tsteps)

    varied_param_mean = np.array([10.5, 10])
    varied_param_cov = np.array([[2, 3], [3, 6]])

    for i in tqdm(range(subjects)):
        varied_param = np.random.multivariate_normal(
            varied_param_mean, varied_param_cov
        )
        print(f"Subject {i + 1}: varied_param = {varied_param}")
        bursting_params = (
            6,
            8,
            -80,
            20,
            60,
            -20,
            15,
            varied_param[0],
            -90,
            -25,
            5,
            1,
            25,
            -15,
            5,
            200,
            10.2,
        )
        lorenz_params = (varied_param[1], 8 / 3, 28)
        for data in [training_data, test_data]:
            bursting_initial_conditions = np.array([-24.46954, 0.00386, 0.0231])
            lorenz_initial_conditions = np.random.rand(3)

            bursting_solution = solve_ivp(
                bursting_neuron,
                [0, t_bursting],
                bursting_initial_conditions,
                args=bursting_params,
                t_eval=bursting_timesteps,
            )
            lorenz_solution = solve_ivp(
                lorenz,
                [0, t_lorenz],
                lorenz_initial_conditions,
                args=lorenz_params,
                t_eval=lorenz_timesteps,
            )
            data[i * 2, :, :] = torch.tensor(bursting_solution.y.T)
            data[i * 2 + 1, :, :] = torch.tensor(lorenz_solution.y.T)

            # Normalize the data
            data[i, :, :] = (data[i, :, :] - data[i, :, :].mean(dim=0)) / data[
                i, :, :
            ].std(dim=0)
            # if data is training_data:
            #    data[i, :, :] += torch.tensor(np.random.normal(0, 0.1, (tsteps, 3)))

    torch.save(
        training_data,
        "data/correlated_system/lorenz_bursting_1vec_correlated_train_reshaped.pt",
    )
    torch.save(
        test_data,
        "data/correlated_system/lorenz_bursting_1vec_correlated_train_reshaped.pt",
    )


def lorenz_roessler_correlated():
    """
    Generates correlated data from the Lorenz and roessler systems.
    Saves the training and test datasets as PyTorch tensors.
    """
    subjects = 50
    tsteps = 100000
    training_data = torch.zeros((subjects * 2, tsteps, 3))
    test_data = torch.zeros((subjects * 2, tsteps, 3))
    t_lorenz = 500
    t_roessloer = 5000
    roessler_timesteps = np.linspace(0, t_roessloer, tsteps)
    lorenz_timesteps = np.linspace(0, t_lorenz, tsteps)

    varied_param_mean = np.array([14, 10])
    varied_param_cov = np.array([[5, 4], [4, 5]])

    for i in tqdm(range(subjects)):
        varied_param = np.random.multivariate_normal(
            varied_param_mean, varied_param_cov
        )
        print(f"Subject {i + 1}: varied_param = {varied_param}")
        roessler_params = (0.2, 0.2, varied_param[0])
        lorenz_params = (varied_param[1], 8 / 3, 28)
        for data in [training_data, test_data]:
            bursting_initial_conditions = np.array([-24.46954, 0.00386, 0.0231])
            lorenz_initial_conditions = np.random.rand(3)

            roessler_solution = solve_ivp(
                roessler,
                [0, t_roessloer],
                bursting_initial_conditions,
                args=roessler_params,
                t_eval=roessler_timesteps,
            )
            lorenz_solution = solve_ivp(
                lorenz,
                [0, t_lorenz],
                lorenz_initial_conditions,
                args=lorenz_params,
                t_eval=lorenz_timesteps,
            )
            data[i * 2, :, :] = torch.tensor(roessler_solution.y.T)
            data[i * 2 + 1, :, :] = torch.tensor(lorenz_solution.y.T)

            # Normalize the data
            data[i, :, :] = (data[i, :, :] - data[i, :, :].mean(dim=0)) / data[
                i, :, :
            ].std(dim=0)
            if data is training_data:
                data[i, :, :] += torch.tensor(np.random.normal(0, 0.1, (tsteps, 3)))

    torch.save(
        training_data,
        "data/correlated_system/lorenz_roessler_1vec_correlated_train_reshaped.pt",
    )
    torch.save(
        test_data,
        "data/correlated_system/lorenz_roessler_1vec_correlated_test_reshaped.pt",
    )


lorenz_bursting_correlated()
