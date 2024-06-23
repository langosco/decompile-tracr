import signal
from decompile_tracr.dataset.data_utils import logger


class SigtermReceivedError(Exception):
    pass


def handle_sigterm(signum, frame):
    logger.info("Received SIGTERM, performing cleanup...")
    raise SigtermReceivedError("Received SIGTERM.")


signal.signal(signal.SIGTERM, handle_sigterm)