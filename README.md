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

## GOSPA Metrics

doi: [10.1109/TSP.2008.920469](https://doi.org/10.1109/TSP.2008.920469)

$$
d_p^{(c,\alpha)}(X,Y) \triangleq \left( \min_{\pi \in \Pi_{|Y|}} \sum_{i=1}^{|X|} d^{(c)}\!\left(x_i, y_{\pi(i)}\right)^p \;+\; \frac{c^p}{\alpha}\bigl(|Y|-|X|\bigr) \right)^{\frac{1}{p}}
$$

Parameters used:

- p = 2
- c = 1

## References (Code)

- https://github.com/schneiderkamplab/bitlinear/tree/main
