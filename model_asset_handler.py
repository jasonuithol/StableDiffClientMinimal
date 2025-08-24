import os
import builtins
import threading
import requests
from constants import get_model_config

def model_asset_handler(model_key: str):

    model_config = get_model_config(model_key)
    MODEL_PATH = model_config["MODEL_PATH"]    
#    LOCAL_MODELS_PATH = model_config["LOCAL_MODELS_PATH"]
    REPOSITORY_ROOT = model_config["REPOSITORY_ROOT"]

    # === CONFIG ===
    MANIFEST_PATH = os.path.join(MODEL_PATH, "_manifest.txt")

    # Keep originals
    _real_open    = builtins.open
    _real_exists  = os.path.exists
    _real_isfile  = os.path.isfile

    _fetch_lock   = threading.Lock()

    # === GLOBAL CACHE OF 404s (persists until process exit) ===
    _missing_files = set()

    # === UTILS ===
    def _should_fetch(path: str) -> bool:
        # Only intercept under LOCAL_MODELS_PATH
        return MODEL_PATH.lower() in os.path.abspath(path).lower()

    def _derive_rel_path(abs_path: str) -> str:
        return os.path.relpath(abs_path, MODEL_PATH).replace("\\", "/")

    # === BLOCKING FETCH (for exists/isfile) ===
    def _fetch_and_cache_blocking(rel_path: str) -> str | None:
        if rel_path in _missing_files:
            # Known ghost, skip fetch entirely
            return None

        local_path = os.path.join(MODEL_PATH, rel_path).replace("\\", "/")
        if _real_exists(local_path):
            return local_path

        with _fetch_lock:
            if not _real_exists(local_path):
                rel_path_for_url = rel_path.replace("\\", "/")
                url = f"{REPOSITORY_ROOT}/{rel_path_for_url}"
                print(f"[file_access_logger] Downloading {url} → {local_path}")

                tmp_path = local_path + ".part"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                r = requests.get(url, stream=True)
                if r.status_code == 404:
                    print(f"[file_access_logger] 404 Not Found - this file doesn't exist anymore. "
                        f"Some old piece of crusty logic needs to be deleted somewhere.")
                    _missing_files.add(rel_path)
                    return None
                r.raise_for_status()
                with _real_open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, local_path)
        return local_path

    # === MANIFEST LOGGING ===
    def _log_manifest(rel_path: str):
        with _fetch_lock:
            with _real_open(MANIFEST_PATH, "a", encoding="utf-8") as mf:
                mf.write(rel_path + "\n")

    # === WRAPPERS ===
    def smart_exists(path) -> bool:
        if path is None or "None.py" in str(path):
            print(f"DEBUG: Blocking ghost fetch (path={path})")
            return False

        path_str = str(path)
        if _real_exists(path_str):
            return True
        if _should_fetch(path_str):
            rel_path = _derive_rel_path(path_str)
            result = _fetch_and_cache_blocking(rel_path)
            return result is not None and _real_exists(result)
        return False

    def smart_isfile(path) -> bool:
        return smart_exists(path) and _real_isfile(path)

    def smart_open(file, mode="r", *args, **kwargs):
        path_str = str(file)

        # Only intercept local model paths
        if "r" in mode and _should_fetch(path_str):
            if not _real_exists(path_str):
                rel_path = _derive_rel_path(path_str)
                if rel_path in _missing_files:
                    raise FileNotFoundError(f"No such file in repo or locally: {path_str}")

                result = _fetch_and_cache_blocking(rel_path)
                if result is None:
                    raise FileNotFoundError(f"No such file in repo or locally: {path_str}")
                # Update path in case fetch adjusted anything (rare, but safe)
                path_str = result

            # Always log manifest once we know the file exists
            if _real_exists(path_str):
                _log_manifest(_derive_rel_path(path_str))

        return _real_open(path_str, mode, *args, **kwargs)

    # === MONKEYPATCH ===
    builtins.open       = smart_open
    os.path.exists      = smart_exists
    os.path.isfile      = smart_isfile