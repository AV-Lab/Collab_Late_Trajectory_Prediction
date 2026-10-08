from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_TMP_DIR = OUTPUTS_DIR / "tmp"
OUTPUTS_PLOTS_DIR = OUTPUTS_DIR / "plots"
MESSAGE_SIZE_TMP_DIR = OUTPUTS_TMP_DIR / "message_size"
MESSAGE_SIZE_PLOT = OUTPUTS_PLOTS_DIR / "message_size.png"
DELAY_TMP_DIR = OUTPUTS_TMP_DIR / "delay"
DELAY_SCATTER_PLOT = OUTPUTS_PLOTS_DIR / "scatter_packet_delay.png"
DELAY_DISTRIBUTION_PLOT = OUTPUTS_PLOTS_DIR / "delay_distribution.png"
RUNTIME_TMP_DIR = OUTPUTS_TMP_DIR / "runtime"
RUNTIME_PLOT = OUTPUTS_PLOTS_DIR / "runtime.png"
CALIBRATION_PLOT = OUTPUTS_PLOTS_DIR / "calibration_curve.png"
