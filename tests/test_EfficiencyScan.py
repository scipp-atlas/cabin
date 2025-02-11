import cabinlearn.EfficiencyScanNetwork as EfficiencyScanNetwork

m = 2
targeteffics = [0.8]
cuts_gt_lt = [1, -1]
activation_input_scale_factor = 15

net = EfficiencyScanNetwork(m, targeteffics, cuts_gt_lt, activation_input_scale_factor)
print(net)
