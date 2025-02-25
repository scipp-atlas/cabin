from __future__ import annotations

import torch


class lossvars:

    def __init__(self):
        self.efficloss = 0
        self.backgloss = 0
        self.cutszloss = 0
        self.monotloss = 0
        self.BCEloss = 0
        self.signaleffic = 0
        self.backgreffic = 0

    def totalloss(self):
        return (
            self.efficloss
            + self.backgloss
            + self.cutszloss
            + self.monotloss
            + self.BCEloss
        )

    def __add__(self, other):
        third = lossvars()
        third.efficloss = self.efficloss + other.efficloss
        third.backgloss = self.backgloss + other.backgloss
        third.cutszloss = self.cutszloss + other.cutszloss
        third.monotloss = self.monotloss + other.monotloss
        third.BCEloss = self.BCEloss + other.BCEloss

        if type(self.signaleffic) is list:
            third.signaleffic = self.signaleffic
            third.signaleffic.append(other.signaleffic)
        else:
            third.signaleffic = []
            third.signaleffic.append(self.signaleffic)
            third.signaleffic.append(other.signaleffic)
        if type(self.backgreffic) is list:
            third.backgreffic = self.backgreffic
            third.backgreffic.append(other.backgreffic)
        else:
            third.backgreffic = []
            third.backgreffic.append(self.backgreffic)
            third.backgreffic.append(other.backgreffic)
        return third


# Basically a more sophisticated version of S/sqrt(B) or S/B.
# see https://cds.cern.ch/record/2736148
def ATLAS_significance_loss(y_pred, y_true, reluncert=0.2, eps=1e-12):
    s = y_pred * y_true
    b = y_pred * (1.0 - y_true)
    
    # Add epsilon for numerical stability
    b_safe = b + eps
    sigma = reluncert * b_safe
    
    # Prevent negative values through clamping
    n = torch.clamp(s + b_safe, min=eps)
    
    # Compute x and y with safe denominators
    sigma_sq = torch.square(sigma)
    denom = torch.clamp(b_safe**2 + n * sigma_sq, min=eps)
    x = n * torch.log((n * (b_safe + sigma_sq)) / denom)
    
    # Compute y term with regularization
    y_num = sigma_sq * (n - b_safe)
    y_denom = torch.clamp(b_safe * (b_safe + sigma_sq), min=eps)
    y = (b_safe**2 / sigma_sq) * torch.log(1 + y_num / y_denom)
    
    # Handle pure signal case (b=0)
    mask = (b < eps) & (sigma < eps)
    x = torch.where(mask, n * torch.log(n/eps), x)
    y = torch.where(mask, n - b_safe, y)
    
    return -torch.sqrt(2 * torch.clamp(x - y, min=eps))



def loss_fn(
    y_pred,
    y_true,
    features,
    net,
    target_signal_efficiency=0.8,
    alpha=1.0,
    beta=1.0,
    gamma=0.001,
    delta=0.0,
    debug=False,
):

    loss = lossvars()

    # signal efficiency: (selected events that are true signal) / (number of true signal)
    signal_results = y_pred * y_true
    loss.signaleffic = torch.sum(signal_results) / torch.sum(y_true)

    # background efficiency: (selected events that are true background) / (number of true background)
    background_results = y_pred * (1.0 - y_true)
    loss.backgreffic = torch.sum(background_results) / (torch.sum(1.0 - y_true))

    # force signal efficiency to converge to a target value. should this be a
    # relative efficiency difference, instead of absolute?  investigate this.
    loss.efficloss = alpha * torch.square(target_signal_efficiency - loss.signaleffic)

    # force background efficiency to small values.  will tend to overweight background
    # effic compared to signal effic since signal effic is squared.  investigate this.
    # generally prefer for loss functions to be sum-of-squares to ensure loss is convex,
    # but in this case the background efficiency is strictly positive, and we want to
    # push it to zero.
    loss.backgloss = beta * loss.backgreffic

    # also prefer to have the cuts be close to zero, so they're not off at some crazy
    # value even if the cut doesn't discriminate much
    cuts = net.get_cuts()
    loss.cutszloss = gamma * torch.sum(torch.square(cuts)) / features

    # calculate the BCE loss, just because.
    loss.BCEloss = delta * torch.nn.BCELoss()(y_pred, y_true)

    if debug:
        print(
            f"Inspecting efficiency loss: alpha={alpha}, target={target_signal_efficiency:4.3f}, subnet_effic={loss.signaleffic:5.4f}, subnet_backg={loss.backgreffic:5.4f}, efficloss={loss.efficloss:4.3e}, backgloss={loss.backgloss:4.3e}"
        )

    return loss


def effic_loss_fn(
    y_pred,
    y_true,
    features,
    net,
    alpha=1.0,
    beta=1.0,
    gamma=0.001,
    delta=0.0,
    epsilon=0.001,
    debug=False,
):

    # probably a better way to do this, but works for now
    sumefficlosses = None
    for i in range(len(net.effics)):
        effic = net.effics[i]
        efficnet = net.nets[i]
        loss_i = loss_fn(
            y_pred[i],
            y_true,
            features,
            efficnet,
            effic,
            alpha,
            beta,
            gamma,
            delta,
            debug,
        )
        if sumefficlosses is None:
            sumefficlosses = loss_i
        else:
            # sumefficlosses=torch.add(sumefficlosses,l)
            sumefficlosses = sumefficlosses + loss_i

    loss = sumefficlosses
    # now set up global penalty for cuts that vary net by net.
    # some options:
    # a. penalize a large range of cut values
    # b. penalize large changes between nearest neighbors
    # c. test for non-monotonicity?
    #
    # go for b for now.
    #

    # For a fancier way to force monotonic behavior, see e.g.
    # https://pypi.org/project/monotonicnetworks/
    #
    # Note that this also has issues since sortedeffics won't necessarily have the same
    # index mapping as 'nets'....  so lots of potential problems here.
    #
    sortedeffics = sorted(net.effics)

    if len(sortedeffics) >= 3:
        featureloss = None
        for i in range(1, len(sortedeffics) - 1):
            cuts_i = net.nets[i].get_cuts()
            cuts_im1 = net.nets[i - 1].get_cuts()
            cuts_ip1 = net.nets[i + 1].get_cuts()

            # calculate distance between cuts.
            # would be better to implement this as some kind of distance away from the region
            # between the two other cuts.
            #
            # maybe some kind of dot product?  think about Ising model.
            #
            # maybe we just do this for the full set of biases, to see how many transitions there are?  no need for a loop?
            #
            # otherwise just implement as a switch that calculates a distance if outside of the range of the two cuts, zero otherwise
            fl = None

            # ------------------------------------------------------------------
            # This method just forces cut i to be in between cut i+1 and cut i-1.
            #
            # add some small term so that when cutrange=0 the loss doesn't become undefined
            cutrange = cuts_ip1 - cuts_im1
            mean = (cuts_ip1 + cuts_im1) / 2.0
            distance_from_mean = cuts_i - mean

            # add some offset to denominator to avoid case where cutrange=0.
            # playing with the exponent doesn't change behavior much.
            # it's important that this term not become too large, otherwise
            # the training won't converge.  just a modest penalty for moving
            # away from the mean should do the trick.
            exponent = 2.0  # if this changes, e.g. to 4, then epsilon will also need to increase
            fl = (distance_from_mean**exponent) / ((cutrange**exponent) + 0.1)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            ## can also do it this way, which just forces all sequential cuts to be similar.
            # fl = torch.pow(cuts_i-cuts_im1,2) + torch.pow(cuts_i-cuts_ip1,2) + torch.pow(cuts_im1-cuts_ip1,2)
            # ------------------------------------------------------------------

            if featureloss is None:
                featureloss = fl
            else:
                featureloss = featureloss + fl

        # need to sum all the contributions to this component of the loss from the different features.
        sumfeaturelosses = torch.sum(featureloss) / (len(sortedeffics) - 2) / features
        loss.monotloss = epsilon * sumfeaturelosses

    return loss
