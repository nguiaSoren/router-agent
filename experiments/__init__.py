"""Offline experiments / launch-prep gates for Token Golf.

Standalone — these modules import the shipped pipeline from ``router_agent.*`` and
never reach into any parent tree. Heavy deps (datasets / pandas) are lazy-imported
inside functions so the package imports cleanly with only the core installed.
"""
