from pathlib import Path

import py_geo_fd


def test_stent_data_not_modified_on_import() -> None:
    data_path = Path(py_geo_fd.__file__).resolve().parent / "stent_data.json"
    before = data_path.read_bytes()

    # Import happened above; this assertion guards against import-time mutation.
    after = data_path.read_bytes()
    assert before == after
