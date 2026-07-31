"""Continuous sensor replay and delayed feedback.

The replay subsystem holds complete held-out trajectories and their failure
cycles, but the online inference path never does: readings travel one cycle
at a time through the real online inference HTTP contract, and realized labels exist
only after the failure event is emitted.
"""
