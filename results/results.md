# Results

This directory contains the exact results reported in the TableShift paper.

The CSV file contained in this directory contains the following columns:
* `task`: The name of the benchmark task.
* `estimator`: The name of the estimator/classifier.
* `in_distribution`: Whether the results for this row are in-distribution (vs. out-of-distribution).
* `test_accuracy`: The test accuracy of the given estimator on the task and dataset split (ID or OOD). Specifically, this is the accuracy of the model with the best ID accuracy after 100 iterations of hyperparameter tuing, following the procedure described in our paper.
* `test_accuracy_clopper_pearson_95%_interval`: The 95% Clopper-PEarson interval for the test accuracy measurement, computed according to the relevant test set size (ID or OOD).