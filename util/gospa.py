import torch
from scipy.optimize import linear_sum_assignment


def filter_alive(
    states: torch.tensor, existence: torch.tensor, existence_threshold=0.6
):
    binary_mask = existence > existence_threshold
    return states[binary_mask]


def gospa(predicted, ground_truth, p=2, c=1.0):
    K = predicted.shape[0] if predicted is not None else 0
    M = ground_truth.shape[0]

    # No predictions and no ground_truths
    if K == 0 and M == 0:
        return torch.tensor(0.0, device=predicted.device)

    # No predictions
    if K == 0:
        # All ground-truth targets are missed
        return torch.tensor((M * (c**p) / 2.0) ** (1 / p), device=ground_truth.device)

    # No Ground Truth
    if M == 0:
        # All predicted targets are false
        return torch.tensor((K * (c**p) / 2.0) ** (1 / p), device=ground_truth.device)

    # Compute pairwise distances (Euclidean)
    dist = torch.cdist(predicted.to(torch.float), ground_truth.to(torch.float), p=2)

    # Apply cutoff c
    cost_matrix = torch.minimum(dist, torch.tensor(c)) ** p

    # Solve optimal assignment using Hungarian
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

    # Matched part
    matched_cost = cost_matrix[row_ind, col_ind].sum()

    # Cardinality mismatch penalty
    missed = M - len(col_ind)
    false = K - len(row_ind)
    cardinality_penalty = ((missed + false) * (c**p)) / 2.0

    return (matched_cost + cardinality_penalty) ** (1 / p)
