# Collab_Late_Trajectory_Prediction
CoLTP framework - Collaborative Multi-Agent Trajectory Prediction with Late Fusion

### Dataset preprocessing

```bash
python -m data.preprocess.preprocess_OPV2V /path/to/OPV2V
python -m data.preprocess.preprocess_DeepAccident /path/to/DeepAccident
python -m data.preprocess.preprocess_V2V4Real /path/to/V2V4Real
```

All three preprocessors write world states in the shared `waymo_v1` convention
with positions in metres, headings in radians and velocities in m/s. OPV2V and
DeepAccident reflect the native world Y axis; V2V4Real keeps its world axes.
LiDAR files and sensor-local axes stay native. DeepAccident recovers the released
object velocity convention and reads ego world velocity from the label header;
V2V4Real derives ego/object velocities from timed world positions before range
filtering. Numeric frame gaps are retained in DeepAccident and V2V4Real.

Reconversion invalidates derived `tmp/vehicle_*.pkl` files so initialization
rebuilds them from the updated dataset.

Trajectory predictors are trained externally. COLTP consumes pretrained predictors for inference.

### Calibration certificates

`calibration/estimators/` contains bootstrap estimation and motion grouping;
`calibration/data/` contains inference and history-error fitting/sampling.
The CLI, runtime lookup and metrics remain at the package root.

Put TrajZoo's `trajzoo-code` on `PYTHONPATH`. Calibration archives and checkpoints
must use the shared Waymo coordinate convention. Observation length, future
horizon and sampling rate come from their metadata; the same commands apply to
OPV2V, V2V4Real and DeepAccident.

Fit either the unbinned category/horizon bootstrap or the motion/covariance
certificate (10 trace-quantile bins by default):

```bash
python -m calibration.certificate fit --method bootstrap --checkpoint /path/to/best.pth --dataset /path/to/heldout.pkl --device cpu --output outputs/calibration/model/bootstrap/certificate.json
python -m calibration.certificate fit --method motion_covariance --cov-bins 10 --checkpoint /path/to/best.pth --dataset /path/to/heldout.pkl --device cpu --output outputs/calibration/model/motion_covariance/certificate.json
```

For Gaussian calibration, first measure paired clean/tracked histories from
held-out physical scenes for that dataset and detector/tracker setup. Each
paired frame contains `contexts.clean`, `contexts.tracked`, and current
tracked-to-clean identity `matches`; future GT is not needed for this step.
The accompanying `history.json` declares the history grid, calibration split,
source identity and detector/tracker settings. The collector also accepts the
existing matched-history captures.

```bash
python -m calibration.data.history_errors --histories /path/to/paired_histories --output data/noise/DATASET/noise_parameters.json
python -m calibration.certificate fit --method motion_covariance --histories gaussian --noise-model data/noise/DATASET/noise_parameters.json --checkpoint /path/to/best.pth --dataset /path/to/heldout.pkl --device cpu --output outputs/calibration/model/motion_covariance_gaussian/certificate.json
```

Replace `DATASET` with `OPV2V`, `V2V4Real`, or `DeepAccident`. The existing OPV2V
parameters are stored at `data/noise/OPV2V/noise_parameters.json`.

Noise parameters retain category/observed-length frequencies, measured bias and
full temporal XY covariance. Sampling changes observed XY, including the current
point, while retaining clean scene membership and GT futures. One joint draw is
used per agent/window. Short histories remain short; motion fitting excludes
windows without a motion tag. Both clean and Gaussian calibration use COLTP's
runtime predictor adapters. Gaussian parameter-fitting uncertainty is outside
the certificate bootstrap.

Every estimator uses 2,000 whole-trajectory bootstrap draws, seed 42, and the
95th percentile spectral-norm deviation of the raw, uncentred error moment.
Bootstrap stores the moment envelope and converts it against each forecast's
covariance into dimensionless factors and then squared-error risks. Motion and
covariance grouping stores direct lower/upper risks in m². Each group needs 100
valid calibration forecasts; sparse covariance bins use their parent
motion/horizon risk. Duplicate quantile boundaries are removed. These are
approximate group-moment estimates; decision-conditioned transfer and zero
cross-source error moments remain assumptions.

Each vehicle loads its own certificate using `predictor.calibration_certificate`
and transmits selected per-step lower/upper risks. All these certificate choices
use `category_l.fusion.type: linear_fusion`: one inverse-upper-risk weight per
source/availability interval and the existing fused-upper < ego-lower gate.
The configured interval-mean covariance-trace filter remains 5 m². Receivers
consume transmitted risks; they do not load peers' certificates.

Evaluate a frozen certificate on separate physical scenarios:

```bash
python -m calibration.certificate evaluate --checkpoint /path/to/best.pth --certificate /path/to/certificate.json --dataset /path/to/independent.pkl --device cpu --output outputs/calibration/model/evaluation.json
```

`--device` selects CPU or a CUDA device; `--batch-size` limits track inference
batches. Scene models preserve their complete scene context. Fitting saves the
certificate and its input provenance without requiring prediction-cache files.
