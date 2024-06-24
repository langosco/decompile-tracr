import signal
from decompile_tracr.dataset.data_utils import logger


class Signals:
    sigterm = False
    
    @classmethod
    def handle_sigterm(cls, signum, frame):
        cls.sigterm = True
        logger.info("Received SIGTERM.")


signal.signal(signal.SIGTERM, Signals.handle_sigterm)