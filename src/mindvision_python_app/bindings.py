import importlib
import os
import sys


def _candidate_release_dirs():
    package_dir = os.path.dirname(__file__)
    project_root = os.path.realpath(os.path.join(package_dir, "..", "..", ".."))

    configured_dir = os.environ.get("MINDVISION_QOBJECT_RELEASE_DIR")
    native_release_dir = os.path.join(project_root, "native", "mindvision_qobject", "release")
    legacy_release_dir = os.path.join(project_root, "..", "mindvision_qobject", "release")

    candidates = []
    for candidate in (configured_dir, native_release_dir, legacy_release_dir):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _load_module():
    import_error = None
    for release_dir in _candidate_release_dirs():
        if not os.path.isdir(release_dir):
            continue
        if release_dir not in sys.path:
            sys.path.insert(0, release_dir)
        try:
            return importlib.import_module("_mindvision_qobject_py")
        except ModuleNotFoundError as exc:
            import_error = exc

    if import_error is not None:
        raise ModuleNotFoundError(
            "Could not import _mindvision_qobject_py. Build the extension in "
            "mindvision_python_app/native/mindvision_qobject or set "
            "MINDVISION_QOBJECT_RELEASE_DIR to an existing release directory."
        ) from import_error

    raise ModuleNotFoundError(
        "Could not locate a MindVision native release directory. Expected "
        "mindvision_python_app/native/mindvision_qobject/release or an override in "
        "MINDVISION_QOBJECT_RELEASE_DIR."
    )


_module = _load_module()

MindVisionCamera = _module.MindVisionCamera
VideoThread = _module.VideoThread
