"""Garmin ingestion layer.

The ingestion layer is deliberately abstracted behind :class:`GarminSource` so the
data backend can be swapped without touching the feature pipelines:

* ``GarminConnectSource`` — the **recommended / production** path, wrapping the
  unofficial ``python-garminconnect`` client. Requires a Garmin Connect account.
* ``SampleDataSource`` — ``DEMO_MODE`` only. Emits compact, illustrative
  Garmin-shaped snapshots so the notebooks run end-to-end without an account.
  This is fixture/illustration data, **not** a physiology simulator, and is not
  the recommended path.

Use :func:`get_source` to obtain the right backend for a notebook.
"""

from ingestion.garmin_client import (
    GarminSource,
    GarminConnectSource,
    SampleDataSource,
    get_source,
)

__all__ = [
    "GarminSource",
    "GarminConnectSource",
    "SampleDataSource",
    "get_source",
]
