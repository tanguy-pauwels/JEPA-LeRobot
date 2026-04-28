# Pipeline Improvement TODO

## 1. Publish lightweight HDF5 previews per shard

Goal: make it possible to validate a converted shard visually without downloading the full shard.

Proposed output layout:

```text
<split>/shard-00000/
  observation_images_exterior_1_left.h5
  preview/
    preview.h5
    preview_manifest.json
```

Suggested behavior:
- sample a small number of episodes per shard after conversion, for example 10 episodes;
- copy those episodes into a tiny `preview.h5` that keeps the same internal schema as the full HDF5 files;
- store the sampled source episode indices and local preview episode mapping in `preview_manifest.json`;
- upload the preview files together with the shard so they are immediately available on Hugging Face.

Why this matters:
- visual validation becomes cheap and fast;
- the preview stays in HDF5, so it tests the converted format directly instead of testing only exported MP4 or PNG artifacts;
- the same visualization tools can be reused with minimal changes.

Open design points:
- choose deterministic sampling for reproducibility, or random sampling with a stored seed;
- decide whether previews should be camera-specific or one preview file per camera;
- decide whether the preview should preserve full resolution or use a smaller image size for faster downloads.


## 2. Add a preview visualization command

Goal: inspect preview shards from local disk or directly from Hugging Face with almost no download cost.

Proposed additions:
- add a script such as `scripts/visualize_hdf5_preview.py`;
- support:
  - local preview paths;
  - manifest-based shard datasets;
  - direct download of only `preview/preview.h5` and `preview_manifest.json` from HF;
- allow random episode selection inside the preview file.

Why this matters:
- the preview workflow is only useful if inspecting it is trivial;
- it should be a one-command sanity check before training.


## 3. Add a post-upload verification mode

Goal: validate a published dataset repo without running a full conversion again.

Proposed command:
- `scripts/verify_published_hdf5.py`

Checks to include:
- manifest consistency;
- remote file presence and expected size checks;
- preview file presence if preview mode is enabled;
- optional download-and-validate of one random shard preview or one full shard file.

Why this matters:
- separates publication verification from conversion;
- gives a clean pre-training checklist.


## 4. Improve startup and progress observability

Goal: make slow phases unambiguous in the terminal.

Suggested improvements:
- log explicit phase names:
  - metadata loading,
  - episode filtering,
  - local file presence checks,
  - source prevalidation,
  - shard plan construction,
  - HDF5 file initialization,
  - video decode plan construction,
  - upload,
  - remote verification;
- print estimated counts before heavy work:
  - number of episodes,
  - number of source parquet files,
  - number of unique MP4 files per camera,
  - number of shards,
  - total expected frames;
- add elapsed time per phase.

Why this matters:
- avoids dead-air in the terminal;
- makes performance bottlenecks easier to identify.


## 5. Add a source-vs-HDF5 spot-check tool

Goal: compare a few random episodes between the source LeRobot dataset and the converted HDF5 output.

Proposed behavior:
- sample N episodes;
- decode the corresponding source frames;
- compare frame counts, episode boundaries, and optionally a few pixel hashes against HDF5 rows;
- emit a compact report.

Why this matters:
- catches conversion regressions earlier;
- gives stronger confidence than shape-only validation.