# Experiment configuration contract

Every experiment added here must specify:

- a unique name and random seed;
- the train, validation, in-distribution test, and OOD test manifests;
- PDE, mesh family, resolution, DG order, flux, time integrator, and CFL range;
- model architecture, input features, loss, optimizer, and stopping rule;
- the validation-only rule used to select the classification threshold;
- the limiter paired with the indicator during solver-in-the-loop evaluation.

Test labels must not be used for model selection or threshold selection. Results should
be aggregated over at least five training seeds and must include both cell-classification
metrics and numerical-solution metrics.

