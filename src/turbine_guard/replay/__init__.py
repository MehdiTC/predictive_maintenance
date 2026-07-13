"""Loop 8 continuous sensor replay and delayed-feedback subsystem.

The replay subsystem holds complete held-out trajectories and their failure
cycles, but the online inference service never does: readings travel one cycle
at a time through the real Loop 7 HTTP contract, and realized labels exist
only after the failure event is emitted.
"""
