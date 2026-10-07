import logging
from datetime import datetime
from pathlib import Path


def configure_logger(root: Path, level: str) -> Path:
    directory = root / 'logs'
    directory.mkdir(exist_ok=True)
    # Microsecond suffix: repeated runs inside the same second must not collide,
    # otherwise an earlier log file would be truncated.
    path = directory / f'{datetime.now():%Y-%m-%d_%H-%M-%S-%f}.log'
    # Never keep a previous run's file handler attached: each run owns its own log.
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - defensive
            pass
    logging.basicConfig(level=getattr(logging, level.upper()),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding='utf-8')], force=True)
    # A previously configured run (or an embedding application) may have disabled
    # logging globally; a fresh run must always record its own messages.
    logging.disable(logging.NOTSET)
    return path
