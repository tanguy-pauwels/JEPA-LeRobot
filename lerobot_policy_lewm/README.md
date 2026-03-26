# lerobot_policy_lewm

Plugin LeRobot pour une policy LeWorldModel (LeWM), en mode behavior cloning direct:

`image -> latent z -> action`

## Commandes

Le runbook exécutable (pré-requis, train, eval offline, dashboard, comparaison sim, troubleshooting) est maintenu ici:

- [COMMANDS.md](/Users/tanguy/projets/roboticArm/lerobot_policy_lewm/COMMANDS.md)

Ce fichier est la source canonique et sera mis à jour à chaque patch.

## Limites v1

- Pas de planning latent (CEM/solver) dans `select_action`.
- Une seule feature visuelle supportée.
- Pas de dépendance runtime à `stable_worldmodel` / `stable_pretraining`.
- La seed par épisode dans `comparison_report.md` est reconstruite (`seed + episode_index`) car `lerobot-eval` ne persiste pas la seed per-episode dans `eval_info.json` pour ce mode.
