import numpy as np
import torch


def save_dataset(
    path,
    final_training_tensor,
    final_mask,
    final_target_tensor,
    final_labels,
    final_sensor_tensor,
    final_unique_id,
    final_life,
):
    data = {
        "final_training_tensor": final_training_tensor.cpu().numpy(),
        "final_mask": final_mask.cpu().numpy(),
        "final_target_tensor": final_target_tensor.cpu().numpy(),
        "final_labels": final_labels.cpu().numpy(),
        "final_sensor_tensor": final_sensor_tensor.cpu().numpy(),
        "final_unique_id": final_unique_id.cpu().numpy(),
        "final_life": final_life.cpu().numpy(),
    }

    np.save(path, data, allow_pickle=True)
    print(f"Saved dataset to {path}")


def load_dataset(path, as_torch=False, device="cpu"):
    data = np.load(path, allow_pickle=True).item()

    if as_torch:
        for k in data:
            data[k] = torch.tensor(data[k], device=device)

    return (
        data["final_training_tensor"],
        data["final_mask"],
        data["final_target_tensor"],
        data["final_labels"],
        data["final_sensor_tensor"],
        data["final_unique_id"],
        data["final_life"],
    )


if __name__ == "__main__":
    (
        final_training_tensor,
        final_mask,
        final_target_tensor,
        final_labels,
        final_sensor_tensor,
        final_unique_id,
        final_life,
    ) = load_dataset("dataset.npy", as_torch=True, device="cuda")
