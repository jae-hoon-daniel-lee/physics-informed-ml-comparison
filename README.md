# Physics-Informed Machine Learning (PIML) Model Comparison & Analysis

This repository explores the performance of Hamiltonian Neural Networks (HNN) and their variants in learning physical laws, specifically focusing on energy conservation and dissipation.

## 🚀 Project Journey
This project started with an in-depth pursuit of the **Hamiltonian-NN family**. During this journey, I expanded the scope to include other architectures like **Neural Symplectic Form**, where I identified and resolved significant underfitting issues and data normalization flaws present in existing open-source implementations. After addressing these challenges, I returned to the HNN family and reorganized them to share on my GitHub repository.

## 📁 Directory Structure & Progress

### 1. [assess-hamiltonian-nn-family](./assess-hamiltonian-nn-family)
- **Focus:** Comprehensive comparison of HNN, D-HNN, D-HNN2, and DGNet.
- **Additional Feature:** **D-HNN2**, custom-designed to output explicit Hamiltonian values, enabling a direct 'Apple-to-Apple' energy landscape comparison across all models.

### 2. [assess-neural-symplectic-form](./assess-neural-symplectic-form)
- **Focus:** Applying Neural Symplectic Form and other models to capture system dynamics.
- **Outcome:** Successfully resolved the underfitting issues found in the "as-is" source codes, enabling a fair comparison of Neural Symplectic Form with other models such as Hamiltonian NN (HNN), Lagrangian NN (LNN), etc.

### 3. [underfit-issue-in-asis-neural_symplectic_form-git-codes](./underfit-issue-in-asis-neural_symplectic_form-git-codes)
- **Status:** Problem Identification
- **Merit:** Various models—including Hamiltonian NN (HNN), Lagrangian (LNN), Neural Symplectic Form (SYM), Skew Matrix Learning (SKEW), and Neural ODE (NODE)—are all included and independently trainable.
- **Issues:** Identified significant performance issues (e.g., underfitting) in HNN and LNN results within existing Neural Symplectic Form repositories.
- **Findings:** Discovered data normalization flaws in the data preparation stage that led to poor training results, preventing a fair comparison between the Neural Symplectic Form and the genuine performance of other models.

## 📄 Documentation
For a detailed narrative of this research journey, motivations, and technical insights, please refer to:
- [Why_I_Did_This_What_I_Learned.pdf](./Why_I_Did_This_What_I_Learned.pdf)






