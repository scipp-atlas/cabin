import torch
import cabin
import time

# --------------------------------------------
# generate some data

N = 20000  # number of points
m = 10  # dimensions

torch.manual_seed(123)
width_scale_factor = 2.0
mean_scale_factor = 1.0

means_widths = torch.randn(m, 2, 2)

x_signal = torch.stack(
    [
        mean_scale_factor * means_widths[i][0][0]
        + width_scale_factor * torch.abs(means_widths[i][0][1]) * torch.randn(N)
        for i in range(m)
    ]
).T

y_signal = torch.ones(N)

x_backgr = torch.stack(
    [
        mean_scale_factor * means_widths[i][1][0]
        + width_scale_factor * torch.abs(means_widths[i][1][1]) * torch.randn(N)
        for i in range(m)
    ]
).T
y_backgr = torch.zeros(N)

x = torch.concatenate((x_signal, x_backgr))
y = torch.concatenate((y_signal, y_backgr))
# --------------------------------------------


# --------------------------------------------
# create a network

gt = 1.0
lt = -1.0
cuts_gt_lt = []  # have network learn values

targeteffics = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# some hyperparameters
learning_rate = 0.5
batch_size = int(len(y) / 1.0)  # only one batch

# how we scale the inputs to the activation function.
# larger values improve the matching of the signal efficiency
# calculated in the loss function and the efficiency that we
# see when applying the cuts.
activation_input_scale_factor = 15

# parameters for the loss functions
alpha = (
    1e1  # scaling factor to tune how important hitting the target signal efficiency is
)
beta = 1e-1  # scaling factor to tune how important background rejection is
gamma = 1e-5  # scaling factor for how aggressively to push the cuts to zero
delta = 1e-3  # scaling factor for how much to use BCE loss to optimize
epsilon = 1e0  # how much to penalize deviations.  as the exponent goes down, this should go down too.

efficnet = cabin.EfficiencyScanNetwork(
    m, targeteffics, cuts_gt_lt, activation_input_scale_factor
)
efficnet_optimizer = torch.optim.SGD(efficnet.parameters(), lr=learning_rate)
# --------------------------------------------


# --------------------------------------------
# train the network

efficnet_losses = []

xy_train = torch.utils.data.TensorDataset(x.float(), y)
loader = torch.utils.data.DataLoader(xy_train, batch_size=batch_size, shuffle=True)

debug = False
epochs = 20

for epoch in range(epochs):
    efficnet.train()
    start_time = time.time()
    for x_batch, y_batch in loader:

        y_pred_efficnet = efficnet(x_batch)
        efficnet_optimizer.zero_grad()
        efficnet_loss = cabin.effic_loss_fn(
            y_pred_efficnet,
            y_batch,
            m,
            efficnet,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            debug=debug,
        )
        efficnet_loss.totalloss().backward()
        efficnet_optimizer.step()

    efficnet_losses.append(efficnet_loss)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Completed epoch {epoch:2d} in {elapsed_time:4.1f}s, loss is {efficnet_loss.totalloss()}"
    )
# --------------------------------------------
