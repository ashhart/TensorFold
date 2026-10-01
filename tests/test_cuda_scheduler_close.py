"""Scheduler.close: the worker stops once idle and lets go of its decoder, so an engine's GPU memory can be freed."""

import gc
import weakref

from tensorfold.cuda.scheduler import Scheduler


class Idle:
    """A decoder with nothing to decode."""

    def live(self):
        return 0

    def round(self):
        return []

    def finish(self, done):
        pass

    def drop(self):
        return []


def test_close_stops_the_worker_and_frees_the_decoder():
    decoder = Idle()
    ref = weakref.ref(decoder)
    scheduler = Scheduler(decoder)
    del decoder
    gc.collect()
    assert ref() is not None                             # the running worker holds it
    scheduler.close()
    assert not scheduler.thread.is_alive()
    gc.collect()
    assert ref() is None
