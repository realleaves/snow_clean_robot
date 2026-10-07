import logging
from datetime import datetime
from pathlib import Path


def configure_logger(root: Path, level: str) -> Path:
    directory = root / 'logs'
    directory.mkdir(exist_ok=True)
    path = directory / f'{datetime.now():%Y-%m-%d_%H-%M-%S}.log'
    logging.basicConfig(level=getattr(logging, level.upper()),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding='utf-8')], force=True)
    return path
