from pathlib import Path
import shutil
from .log import get_logger

logger = get_logger(__name__)

def create_dirs(list_path_to_dir, list_remove_existing):
    for path_to_dir, remove_existing in zip(list_remove_existing, list_remove_existing):
        create_dir(path_to_dir, remove_existing)

def create_dir(path_to_dir, remove_existing):
    """Creates a new directory at path."""
    path_to_dir = Path(path_to_dir)

    if remove_existing and path_to_dir.exists():
        shutil.rmtree(path_to_dir)
        logger.warning(f"Deleted existing directory at {path_to_dir}")

    if not path_to_dir.exists():
        path_to_dir.mkdir(parents=True)
        logger.info(f"Created new directory at {path_to_dir}")
    else:
        logger.info(f"Using existing directory at {path_to_dir}")
    
    return
