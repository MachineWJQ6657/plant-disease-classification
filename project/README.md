# Tomato classification with fresh views and three paired seeds

Run commands here from **project/**. Final materials are in **../submission/**; the Chinese guide is **../docs/START_HERE_ZH.txt**.

The main experiment now consists of six independently trained models: seeds 664, 665 and 666, ordinary versus strong crop/color augmentation, eight epochs each with newly sampled views every epoch. The original supplied two-view experiment remains separately in results/. The new comparison gives a modest mean field gain, with inconsistent benefits across seeds.

| Three-seed mean ± sample SD (%) | Ordinary | Crop + color |
| --- | ---: | ---: |
| Clean accuracy | 97.26 ± 0.16 | 96.66 ± 0.58 |
| Field accuracy | 18.78 ± 1.10 | 20.13 ± 0.74 |
| Clean macro-F1 | 96.32 ± 0.18 | 95.71 ± 0.50 |
| Field macro-F1 | 7.76 ± 0.83 | 9.16 ± 0.73 |

The paired field accuracy changes are −0.27, +3.10 and +1.21 percentage points. The mean change is +1.35 ± 1.69 pp; macro-F1 changes by +1.40 ± 1.31 pp. Two seeds improve and one declines slightly. These are three fits on the same 742 images, not three independent test collections.

## Open the executed Notebook

**../submission/Tomato_Project.ipynb** contains the real methods, complete recorded training histories, independent metric and bootstrap recomputation, per-class gaps and actual new-checkpoint inference. All 21 code cells executed successfully; outputs are saved. Default execution does not repeat training or download raw data. Heavy operations require explicit False-to-True switches. Viewing the notebook alone needs no raw dataset; rerunning needs the complete package.

## Install a CPU display environment

Use 64-bit Python 3.12. Keep the display environment separate from the GPU training environment below.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv-display
.\.venv-display\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.12 -m venv .venv-display
source .venv-display/bin/activate
```

Linux/Windows CPU installation:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements-notebook.txt
python -m pip check
cd ..
python -m notebook submission/Tomato_Project.ipynb
```

For macOS use the default PyPI index for torch and torchvision. The display environment was verified on Windows CPU and can read both legacy and fresh checkpoints. It is not the environment used for the new GPU training. VS Code can open the notebook with this same interpreter.

## Offline demonstration and saved-evidence audits

After installation, no network or raw dataset is required:

```bash
python demo.py examples/field.jpg
python demo.py examples/field_second.jpg
python verify.py --out results/fresh_views/seed_664
python verify.py --out results/fresh_views/seed_665
python verify.py --out results/fresh_views/seed_666
python summarize_fresh.py --out results/fresh_views
python -m unittest discover -s tests -v
```

The demo defaults to the fixed new seed-664 pair, not the best field seed. Both models call field.jpg late blight; field_second.jpg is early blight for ordinary and late blight for strong. Both dataset labels are early blight. Softmax scores are uncalibrated. run_demo.ps1 / run_demo.sh execute all three audits and both examples. To use legacy weights explicitly, add `--results results`.

verify.py itself needs only the Python standard library; the complete ten-test suite includes fresh-transform tests and requires the installed ML packages. The tests reject missing predictions, corrupt probabilities, cross-split hashes and wrong summary scores, and verify fresh, replayable, thread-independent transformations.

## Reproduce the new training

Create a separate Python 3.12 environment. For the verified CUDA 12.8 build on a compatible NVIDIA GPU:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-train.txt
python -m pip check
python download.py --data data
python experiment_fresh.py --data data --manifest results/manifest.csv --out rerun/fresh_views --cache .build/fresh-cache --device cuda --seeds 664 665 666 --workers 4 --threads 4
python summarize_fresh.py --out rerun/fresh_views
```

Use the [official matched PyTorch builds](https://pytorch.org/get-started/previous-versions/#v271) for other platforms. CPU execution is supported with --device cpu but will differ in speed and potentially floating-point results. The actual new runs used one RTX 5070 Ti Laptop GPU, PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128. Exact installed packages are recorded in results/fresh_views/training_environment.txt. train_longleaf.sbatch requests one GPU and runs the fresh three-seed experiment; supply your site's partition/account and activate the training environment first. That job template was not submitted to Longleaf.

All six models start from ImageNet, not the old fine-tuned weights. Within each seed, initialization and sample order match. Each model gets 1,008 updates and 96,288 image presentations. The prefix remains frozen in evaluation mode; the final residual block, final convolution and classifier train in float32. Prefix activations are cast through float16 as in the legacy method. Only deterministic RGB resize is cached; per-image crop/color/flip parameters are redrawn each epoch with local RNGs independent of thread scheduling.

The protocol and source hash are written before training. Losses, validation scores, step counts and sample-order hashes are recorded each epoch. All six checkpoints are locked by hash before any test scoring. Raw image hashes are checked while constructing pixel caches. Completed conditions and per-epoch optimizer/RNG checkpoints support resuming an interrupted run in the same output/cache paths, provided the source and environment are unchanged. Use a new output directory for a changed protocol.

## Data and inference reproduction

Allow approximately 3.5 GB for pinned archives plus several GB for extracted images. Raw data are not bundled. Download and reconstruct with:

```bash
python download.py --data data
python prepare.py --data data --out reconstructed
```

The reconstructed manifest.csv, exclusions.csv and class_mapping.csv were byte-identical to results/. paths.py reversibly encodes illegal Windows filename characters and supports extended-length paths; manifests keep upstream names. Use disk_path() to access them. Use a fresh extraction directory instead of mixing old unencoded files.

For raw-image replay, use the matching training environment, point --results at one seed and keep --out separate:

```bash
python evaluate.py --data data --results results/fresh_views/seed_664 --out replay/seed_664 --device cuda --compare
```

Repeat for seeds 665 and 666. Independent post-training replay was performed for all six new models: all 24,558 labels match, with maximum probability difference 0.0003113. The replayer verifies every raw test hash and compares checkpoint probabilities and labels. It automatically restores the recorded deterministic and TF32 settings for fresh checkpoints; an initial unconfigured replay and its correction are documented in results/fresh_views/replay_precision_diagnostic.json. Neither weights nor primary results were changed to resolve the numerical mismatch. Hardware/library differences may require reviewing a replay mismatch instead of silently replacing the evidence. The original supplied pair was also independently replayed on all 4,093 raw test images: all 8,186 labels matched, maximum probability difference 0.0002593. That verified inference, not its full optimization history. The fresh run independently executed the revised six-model training.

## Evidence and source files

| Path | Contents |
| --- | --- |
| experiment_fresh.py | Actual fresh-view training, per-epoch logging, selection lock and test evaluation |
| summarize_fresh.py | Paired-budget checks, per-seed/mean/SD aggregation and class gaps |
| results/fresh_views/protocol.json | Frozen seeds, transformations, hyperparameters, environment and script/manifest hashes |
| results/fresh_views/events.jsonl | Timestamped training and evaluation events |
| results/fresh_views/selection_locked.json | All six checkpoint hashes finalized before test scoring |
| results/fresh_views/seed_664, seed_665, seed_666 | Two weights each, histories, completion records, test probabilities, metrics and audits |
| results/fresh_views/summary.csv | Mean and sample SD across three seeds |
| results/fresh_views/paired_seed_deltas.csv | Every paired effect, including the negative seed |
| results/fresh_views/class_summary.csv | Class recall/F1 means and SD; per-seed counterparts are also included |
| results/fresh_views/aggregate_verification.json | Completed budgets, pairing and all per-seed audit checks |
| results/ordinary.pt, strong.pt and original CSVs | Unchanged supplied legacy experiment |
| prepare.py, paths.py, download.py | Audited mapping/split, portable paths and pinned data retrieval |

Original prepare.py, experiment.py, analyze.py and evaluate.py remain the legacy pipeline and shared implementation. New methods are not silently substituted into old result tables.

## Rebuild the report and slides

```bash
python make_fresh_figures.py --results results/fresh_views --dest report/figures
```

Then run report/build_report.ps1 or report/build_report.sh. The scripts compile LaTeX twice into .build/report/ and update ../submission/Tomato_Report.pdf. Source and generated tables/figures are included. The final report is seven pages with resolved references and no overfull boxes.

../submission/Tomato_Slides.pptx has 12 plain white academic slides, editable data tables and speaker notes; a PDF viewing copy is alongside it. Refresh slide_data.json with slides/prepare_slide_data.py. slides/build_slides.mjs and presentation.json are the authoring sources; JavaScript rebuilding requires Codex's bundled @oai/artifact-tool. Manual PowerPoint editing requires no such package. The original two-image credits are in examples/credits.csv.

## Interpretation and remaining limits

Fresh crop/color augmentation improves mean field metrics, but the benefits vary across the three seeds and absolute field performance remains weak. Validation still omits mosaic virus. Exactly 4,280 source-training and 1,060 clean-test images lack documented leaf IDs. Exact-byte audit cannot rule out unknown same-leaf images or near-duplicates. Only two related spider-mite field entries exist. Three seeds characterize limited optimization variability; bootstrap intervals condition on fitted models and observed test images.

The follow-up was designed after inspecting the legacy field results. Field labels are excluded from optimization and checkpoint selection, but this is exploratory evidence on an already examined benchmark. A new independently labeled field collection and predefined target-development/test split are higher priorities than adding unrelated architectures. The old/new absolute difference is not a pure view-refresh ablation because the RNG implementation, data streaming and execution environment also changed.

## Sources

- Mohanty, Hughes & Salathé (2016), [PlantVillage](https://doi.org/10.3389/fpls.2016.01419), [repository](https://github.com/spMohanty/PlantVillage-Dataset), [official grouped split files](https://huggingface.co/datasets/mohanty/PlantVillage/tree/main/splits); the dataset card identifies CC BY-SA 3.0.
- Singh et al. (2020), [PlantDoc](https://doi.org/10.1145/3371158.3371196), [cropped classification repository](https://github.com/pratikkayal/PlantDoc-Dataset), CC BY 4.0.
- Howard et al. (2019), [MobileNetV3](https://arxiv.org/abs/1905.02244).

Third-party data and model assets retain their upstream terms. The report uses the explicitly requested Author Name(s) placeholder; replace it with actual authors when finalizing the submission.

Unix shell entry points use LF, enforced by .gitattributes. Invoke them with bash; the ZIP also records executable permission bits.
