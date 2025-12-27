# Physics-Informed Machine Learning (PIML) Model Comparison

This repository compares the performance of Hamiltonian Neural Networks (HNN) and their variants in learning physical laws.

## 📝 How it proceeded
This work started with an in-depth look into the **Hamiltonian-NN family**. Along the way, I looked into other architectures like **Neural Symplectic Form**, where I found some issues with underfitting and data normalization in existing open-source codes. After dealing with those issues, I returned to the HNN family and reorganized the codes to share here.

## 📁 Directory Structure & Progress

### 1. [assess-hamiltonian-nn-family](./assess-hamiltonian-nn-family)
- **Focus:** Comparison of HNN, D-HNN, D-HNN2, and DGNet.
- **Additional Feature:** D-HNN2, modified to output explicit Hamiltonian values to enable direct energy comparison across the models.

### 2. [assess-neural-symplectic-form](./assess-neural-symplectic-form)
- **Focus:** Applying Neural Symplectic Form and other models to capture system dynamics.
- **Outcome:** Resolved the underfitting issues found in the "as-is" source codes, enabling a fair comparison of Neural Symplectic Form with other models such as HNN, LNN, etc.

### 3. [underfit-issue-in-asis-neural_symplectic_form-git-codes](./underfit-issue-in-asis-neural_symplectic_form-git-codes)
- **Status:** Problem Identification
- **Merit:** Various models—including HNN, LNN, SYM, SKEW, and NODE—are all included and independently trainable.
- **Issues:** Found performance issues (e.g., underfitting) in HNN and LNN results within existing Neural Symplectic Form repositories.
- **Findings:** Discovered data normalization flaws in the data preparation stage, hindering a fair comparison between the models.

## 📄 Documentation
For a detailed overview of this work, including the motivation, theoretical consideration, and lessons learned, please refer to:
- [Why_I_Did_This_What_I_Learned.pdf](./Why_I_Did_This_What_I_Learned.pdf)


















