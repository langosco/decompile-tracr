import signal
from decompile_tracr.dataset.data_utils import logger


class Signals:
    sigterm = False
    n_sigterms = 0
    
    @classmethod
    def handle_sigterm(cls, signum, frame):
        cls.sigterm = True
        cls.n_sigterms += 1
        logger.info("Received SIGTERM.")
        if cls.n_sigterms > 1:
            logger.info(f"Received > 1 SIGTERM ({cls.n_sigterms}).")


signal.signal(signal.SIGTERM, Signals.handle_sigterm)