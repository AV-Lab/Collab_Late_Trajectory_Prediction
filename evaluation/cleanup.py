from evaluation.paths import DELAY_TMP_DIR, MESSAGE_SIZE_TMP_DIR, RUNTIME_TMP_DIR


def clear_runtime_stats():
    for directory in (MESSAGE_SIZE_TMP_DIR, DELAY_TMP_DIR, RUNTIME_TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_path.unlink()
