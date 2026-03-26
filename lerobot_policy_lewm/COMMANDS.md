# COMMANDS.md

## Dernière mise à jour
- Date: 2026-03-26
- Patch: V2 JEPA (triple loss action+dynamics+SIGReg)

## Pré-requis

```bash
# 1) Python 3.12 recommandé (important pour la compat Lerobot locale)
python3.12 -m venv .venv
source .venv/bin/activate

# 2) Installer le plugin en editable
cd /Users/tanguy/projets/roboticArm/lerobot_policy_lewm
pip install -e .

# 3) Vérifier que LeRobot local est utilisé
export PYTHONPATH=/Users/tanguy/projets/roboticArm/lerobot/src
```

## Train LeWM

```bash
cd /Users/tanguy/projets/roboticArm/lerobot_policy_lewm
PYTHONPATH=/Users/tanguy/projets/roboticArm/lerobot/src \
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human_image \
  --policy.type=lewm \
  --policy.push_to_hub=false \
  --policy.device=cpu \
  --policy.use_world_model_loss=true \
  --policy.loss_weight_action=1.0 \
  --policy.loss_weight_dynamics=1.0 \
  --policy.loss_weight_sigreg=0.09 \
  --steps=1000
```

## Overfit sanity (v2)

Objectif: valider rapidement la baisse de `loss_action` et `loss_dynamics` sur un run court.

```bash
cd /Users/tanguy/projets/roboticArm/lerobot_policy_lewm
PYTHONPATH=/Users/tanguy/projets/roboticArm/lerobot/src \
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human_image \
  --policy.type=lewm \
  --policy.push_to_hub=false \
  --policy.device=cpu \
  --batch_size=1 \
  --steps=100 \
  --save_freq=100 \
  --log_freq=5
```

## Eval offline

```bash
cd /Users/tanguy/projets/roboticArm/lerobot_policy_lewm
PYTHONPATH=/Users/tanguy/projets/roboticArm/lerobot/src \
python -m lerobot_policy_lewm.eval_offline \
  --policy-path /Users/tanguy/projets/roboticArm/lerobot_policy_lewm/outputs/train/2026-03-26/12-19-51_lewm/checkpoints/last/pretrained_model/ \
  --dataset-repo-id lerobot/aloha_sim_insertion_human_image \
  --dataset-root /Users/tanguy/projets/roboticArm/datasets/aloha_sim_insertion_human_image \
  --batch-size 1 \
  --max-batches 100 \
  --device cpu \
  --output-dir /tmp/lewm_offline_debug
```

Le script résout automatiquement le dernier checkpoint numérique sous `checkpoints/`.

Artefacts attendus:
- `offline_summary.json`
- `per_dim_metrics.csv`
- `per_episode_metrics.csv`
- `samples_predictions.csv`

## Dashboard

```bash
cd /Users/tanguy/projets/roboticArm/lerobot_policy_lewm
python -m lerobot_policy_lewm.debug_dashboard \
  --artifacts-dir /tmp/lewm_offline_debug \
  --host 127.0.0.1 \
  --port 7860
```

## Check-list d'acceptation offline

- `global_mse` diminue entre checkpoint early et later.
- `global_mae` diminue entre checkpoint early et later.
- `pred_std` n'est pas proche de zéro.
- Dans le dashboard: trajectoires prédictives non constantes et scatter `pred vs target` moins erratique.

## (Optionnel) Comparaison sim

A utiliser seulement après validation offline (MSE/MAE en baisse entre checkpoints).

```bash
cd /Users/tanguy/projets/roboticArm/lerobot_policy_lewm
python -m lerobot_policy_lewm.compare_policies \
  --policy-a-path /path/to/lewm_checkpoint/pretrained_model \
  --policy-b-path /path/to/baseline_checkpoint/pretrained_model \
  --label-a lewm \
  --label-b baseline \
  --env-type aloha \
  --env-task AlohaInsertion-v0 \
  --n-episodes 20 \
  --batch-size 4 \
  --seed 1000 \
  --device cpu \
  --output-root comparison_runs \
  --lerobot-src /Users/tanguy/projets/roboticArm/lerobot/src
```

## Troubleshooting

- `SyntaxError` sur `deserialize_json_into_object[...]`:
  - utiliser Python 3.12 (pas 3.10).
- `ModuleNotFoundError: gym_aloha`:
  - installer les dépendances aloha dans le même venv que `lerobot_policy_lewm`.
- `config.json not found` sur baseline ACT Hub:
  - fournir un vrai checkpoint `pretrained_model` local ou un repo model HF contenant `config.json`.
