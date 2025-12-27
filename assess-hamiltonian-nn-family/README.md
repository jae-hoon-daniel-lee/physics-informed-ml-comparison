# Comparing Hamiltonian-based Neural Networks

This directory is for comparing various Hamiltonian-NN architectures. It currently shows a comparison using real-world data, and I plan to add some synthetic cases later.

## 💡 The 'experiment-real' example
The **'experiment-real'** case is a good example where HNN, D-HNN, and DGNet can all be compared together.

To compare their energy values directly, I used a **slightly modified version of D-HNN** (referred to as D-HNN2 in the code). While the original D-HNN only outputs vector fields, I modified it to output explicit Hamiltonian values, making it possible to compare the energy results across the models on the same basis.

## 📊 Comparison results
The results for the **'experiment-real'** case are shown in the 2x2 subplot below:

![HNN-Family Comparison](./experiment-real/compared/danieljh_plot_real_compared.png)

*The figure shows how HNN, D-HNN, D-HNN2, and DGNet interpret the dynamics and energy of the real-world pendulum.*

## 🚀 Future updates
The following cases are being organized and will be added later. These primarily focus on the comparison between HNN and DGNet:
- **experiment-2body/**: Orbit and energy conservation tests on the 2-body problem.
- **experiment-pend/**: Benchmark on the ideal (frictionless) pendulum system.

## 📂 Weights & Data
- **'experiment-real'** weights: `experiment-real/weights/`
- **'experiment-real'** dataset: `experiment-real/invar_datasets.zip`


