import jax
jax.config.update("jax_enable_x64", False)
import os
os.environ["KERAS_BACKEND"] = "jax"
import bayesflow as bf

workflow = bf.BasicWorkflow(
    inference_network=bf.networks.FlowMatching(),
    inference_variables=["parameters"],
    inference_conditions=["observables"],
    simulator=bf.simulators.SIR()
)

history = workflow.fit_online(epochs=20, batch_size=32, num_batches_per_epoch=200)

diagnostics = workflow.plot_default_diagnostics(test_data=100)