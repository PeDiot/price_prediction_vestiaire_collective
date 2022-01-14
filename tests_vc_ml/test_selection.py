"""Description.

Test model selection functions from the vc_ml library.
"""

from importlib.resources import path
import pytest

from os import remove

from vc_ml import (
    ModelDir, 
    BACKUP, 
    get_files_path, 
    get_cv_results,
    get_best_estimator, 
    save_best_estimator,
    load_best_estimator
)

def test_model_dir():
    dir = ModelDir.GB 
    assert isinstance(dir, ModelDir)
    assert dir.value == "GradientBoostingRegressor/"

def test_error_get_files(): 
    """Create an example file which is not pickle."""
    dir = ModelDir.GB
    file_path = BACKUP + "models/" + dir.value + "demofile.txt"
    f = open(file_path, "w")
    f.write("Not pickle file.")
    f.close()
    with pytest.raises(ValueError):
        paths = get_files_path()
    remove(path=file_path)

    


