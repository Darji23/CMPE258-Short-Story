# Short Story Assignment: Review & Reproduction of "GraphAny"

**Topic:** Fully-inductive Node Classification on Arbitrary Graphs  
**Original Paper:** [https://arxiv.org/abs/2405.10000 ](https://arxiv.org/abs/2405.20445) 

---

## 📌 Deliverables

> **Note to Reviewer**: The following are placeholders for the final submission links.

- **Medium Article**: [Link to Medium Article](#)
- **Slide Deck**: [Link to SlideShare / Google Slides](#)
- **YouTube Video Presentation**: [Link to YouTube Video](#) (15-25 minutes explaining the paper and code)

---

## 📖 Paper Summary: GraphAny

### 1. Introduction & Motivation
Standard Graph Neural Networks (GNNs) typically operate under a *transductive* setting, meaning they require the entire graph topology (both training and test nodes) to be present during training. Even standard *inductive* models usually assume that the target graph shares the same feature space and label space as the training graph. 

**GraphAny** introduces a groundbreaking **fully-inductive** paradigm. It allows a model trained on one graph (e.g., the `Wisconsin` dataset) to be evaluated on a completely different graph (e.g., the `Texas` dataset) without any retraining, fine-tuning, or alignment of node feature dimensions or label spaces.

### 2. Core Architecture
GraphAny achieves this generalizability through a clever combination of **LinearGNNs** and **Entropy-Normalized Distance Features**.

#### A. LinearGNNs with Analytical Solvers
Instead of learning complex, parameterized convolutions via backpropagation, GraphAny uses purely analytical LinearGNNs. It generates multiple filtered feature representations (e.g., $X, A X, A^2 X, (I-A)X$). For each filter $F_L$, the optimal weight matrix $W^*$ is computed analytically using the Moore-Penrose pseudo-inverse on the training labels $Y_L$:
$$ W^* = F_L^+ Y_L $$
This allows the model to produce initial predictions efficiently without gradient descent.

#### B. Entropy-Normalized Distance Features
Since different graphs have different numbers of classes, the output dimension of the LinearGNNs changes. To make the attention mechanism invariant to the number of classes, GraphAny computes the **pairwise distances** between the predictions of the different LinearGNNs. These distances are then *entropy-normalized* (similar to the conditional Gaussian probabilities used in t-SNE) to ensure the distance distributions remain stable across different graphs.

#### C. Inductive Attention
An attention MLP takes these scale-invariant, graph-agnostic distance features and outputs attention weights. These weights are used to combine the predictions of the various LinearGNNs into a final, robust prediction. Because the attention mechanism only looks at *distances between predictions* rather than the raw features or labels, it can transfer perfectly to unseen graphs.

### 3. Key Findings & Ablation Studies
The paper evaluates GraphAny across 31 diverse datasets. Key findings include:
- **Zero-Shot Superiority**: GraphAny consistently outperforms heavily parameterized models (like GCN, GAT) in cross-graph zero-shot transfer.
- **Attention Robustness**: The entropy-normalized attention effectively learns to trust High-Pass filters for heterophilous graphs and Low-Pass filters for homophilous graphs automatically.

---

## 🛠 Implementation Details & Reproduction

As part of this assignment, the GraphAny model was successfully reproduced using an `autoresearch` template philosophy. The model was trained on the **Wisconsin** dataset and tested zero-shot on the **Texas** dataset.

### Project Structure

```text
├── GraphAny/                   # Core implementation
│   ├── configs/                # Hydra configs (datasets, hyperparameters)
│   ├── graphany/
│   │   ├── data.py             # PyTorch Geometric data processing & message passing
│   │   ├── model.py            # LinearGNN & Attention architecture
│   │   └── run.py              # Main training loop (PyTorch Lightning)
├── output/                     # Generated evaluation metrics & plots
│   ├── metrics.json            # Final accuracy metrics
│   └── loss_curve.png          # Visualizations
├── venv/                       # Virtual environment (dependencies)
└── README.md                   # Project documentation
```

### Setup Instructions

The reproduction bypasses legacy constraints (like DGL dependencies on certain machines) by leveraging robust native PyTorch Geometric sparse matrix multiplications.

**1. Create Environment & Install Dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio torch_geometric pytorch-lightning lightning
pip install pydantic wandb rich hydra-core einops ogb rootutils scikit-learn scipy matplotlib
```

**2. Execute the Reproduction:**
```bash
cd GraphAny
source ../venv/bin/activate
python graphany/run.py
```

---

## 📊 Reproduction Results

The reproduction successfully mirrors the paper's fully-inductive claims. Training was completed exclusively on the `Wisconsin` graph, and the model generalized to the `Texas` graph.

| Evaluation | Accuracy |
|------------|----------|
| **Validation (Wisconsin)** | ~72.88% |
| **Zero-Shot Test (Texas)** | ~70.27% |

The strong 70%+ accuracy on Texas, despite differing feature dimensions, edge densities, and label classes from Wisconsin, empirically validates GraphAny's entropy-normalized inductive attention framework.

---

## 📚 References

- **Reference Codebase:** [DeepGraphLearning/GraphAny](https://github.com/DeepGraphLearning/GraphAny)
- **Libraries used:** PyTorch, PyTorch Geometric, PyTorch Lightning, Hydra.
