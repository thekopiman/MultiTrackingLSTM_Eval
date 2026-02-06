# MultiTrackingLSTM_Eval

Simulation and Evaluation npy used for Benchmarking

Code for the model will not be provided.

## Loading the dataset

```
(
        final_training_tensor,
        final_mask,
        final_target_tensor,
        final_labels,
        final_sensor_tensor,
        final_unique_id,
        final_life,
    ) = load_dataset(test/test_50.npy, as_torch=True, device="cuda")
```

## References (Code)

- https://github.com/schneiderkamplab/bitlinear/tree/main
