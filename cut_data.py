import torch

train_data = torch.load(
    "./data/correlated_system/lorenz_bursting_1vec_correlated_train.pt"
)

test_data = torch.load(
    "./data/correlated_system/lorenz_bursting_1vec_correlated_test.pt"
)

# torch.save(
#    train_data[:, :, :3],
#    "./data/correlated_system/lorenz_bursting_1vec_correlated_train_cut.pt",
# )
# torch.save(
#    test_data[:, :, :3],
#    "./data/correlated_system/lorenz_bursting_1vec_correlated_test_cut.pt",
# )

"""
torch.save(
    torch.cat((train_data[:, :, :3], train_data[:, :, 3:]), axis=0),
    "./data/correlated_system/lorenz_bursting_1vec_correlated_train_reshaped.pt",
)
torch.save(
    torch.cat((test_data[:, :, :3], test_data[:, :, 3:]), axis=0),
    "./data/correlated_system/lorenz_bursting_1vec_correlated_test_reshaped.pt",
)
"""
