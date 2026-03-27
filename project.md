# POC : Intégration de LeWorldModel (JEPA) dans LeRobot (2026)

## 🔗 Références
- **World Model (LeWM)** : https://github.com/lucas-maes/le-wm (Architecture JEPA + SIGReg)
- **LeRobot (HF)** : https://github.com/huggingface/lerobot (Framework Robotique)

## 1. Core Objective
L'objectif de ce projet est d'implémenter une architecture **JEPA (Joint-Embedding Predictive Architecture)** pour le contrôle robotique au sein de l'écosystème **LeRobot**. Contrairement au *Behavioral Cloning* classique, **LeWM** sépare l'apprentissage de la physique du monde (World Model) de la prise de décision (Planning).

## 2. System Architecture

### 🧠 World Model (The "Brain")
* **Vision Encoder :** ViT-Tiny (Hugging Face) avec normalisation **ImageNet** ($mean=[0.485, 0.456, 0.406]$, $std=[0.229, 0.224, 0.225]$).
* **Latent Space ($Z$) :** Espace compressé de dimension `embed_dim=192`.
* **Predictor :** Modèle auto-régressif (Transformer/MLP) prédisant $\hat{z}_{t+1}$ à partir de l'état actuel $z_t$ et de l'action $a_t$.
* **SIGReg :** Régularisation *Isotropic Gaussian* pour maintenir la structure de l'espace latent et éviter l'effondrement (mode collapse).

### 🎯 Inference & Control (The "Solver")
* **Goal-Conditioning :** Le système est guidé par un embedding cible $z_{goal}$ issu d'une image de succès.
* **CEM (Cross-Entropy Method) :** Algorithme d'optimisation itératif qui cherche la meilleure séquence d'actions dans l'espace imaginaire du Predictor.
* **MPC (Model Predictive Control) :** Planification sur un horizon $H$ (ex: 15 steps) avec exécution partielle ($k$ steps) pour assurer une boucle de rétroaction visuelle robuste.

---

## 3. Training Protocol

### Phase 1 : Self-Supervised Dynamics
Apprentissage des transitions latentes sans tête d'action explicite.
* **Dynamics Loss :** $$L_{dyn} = \| \text{Predictor}(z_t, a_t) - \text{detach}(z_{t+1}) \|^2$$
* **Regularization Loss :** $$L_{sigreg}$$ (via Sketch Isotropic Gaussian).

### Phase 2 : Planning Validation (Offline)
Validation de la capacité du solveur CEM à trouver des trajectoires menant au $z_{goal}$ dans le dataset de test.

---

## 4. Dataset & Hardware Configuration
* **Robot :** Bras Koch (6-DOF)
* **Dataset :** `lerobot/koch_pick_place_5_lego` (Statique).
* **Pipeline :** Conversion Parquet/MP4 (LeRobot) vers HDF5 (LeWM Core) pour l'entraînement intensif des dynamiques.

---

## 5. Implementation Status & Roadmap
- [] Script de convertion Parquet/MP4 (LeRobot) vers HDF5 (LeWM Core) pour l'entraînement intensif des dynamiques.
- [] Entrainement du World Model sur le dataset converti
- [] Intégration de la recherche de Goal Embedding 
- [] Intégration du World Model
- [] Intégration du solveur CEM
- [] Intégration du solveur MPC
- [] Validation fonctionnelle sur une tâche de manipulation simple (Pick & Place)

---

### 💡 Documentation & Usage


---
