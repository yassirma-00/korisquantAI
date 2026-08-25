"""JSON response class that survives real-world numeric payloads.

Financial analytics routinely produce ``NaN`` / ``±Inf`` (a rolling indicator
before its warm-up period, a ratio with a zero denominator, ...), plus numpy
scalars and pandas timestamps. Standard ``json.dumps`` refuses NaN and chokes
on numpy types, so every endpoint would need manual sanitising.

This response class normalises everything once, centrally:

* ``NaN`` / ``Inf``      -> ``null``
* numpy scalars / arrays -> Python scalars / lists
* pandas Timestamp / NaT -> ISO string / ``null``
* ``Decimal``, ``set``, dataclasses -> JSON-friendly equivalents
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse


def sanitise(obj: Any) -> Any:
    """Recursively convert *obj* into something ``json.dumps`` accepts."""
    # -- scalars that are already fine -----------------------------------
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    # -- floats (catch NaN / Inf) ----------------------------------------
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, int):
        return obj

    # -- numpy -----------------------------------------------------------
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return [sanitise(v) for v in obj.tolist()]

    # -- pandas ----------------------------------------------------------
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return [sanitise(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [sanitise(rec) for rec in obj.to_dict(orient="records")]

    # -- datetime / decimal ----------------------------------------------
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if isinstance(obj, dt.timedelta):
        return obj.total_seconds()
    if isinstance(obj, Decimal):
        value = float(obj)
        return value if math.isfinite(value) else None

    # -- containers ------------------------------------------------------
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(k)): sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [sanitise(v) for v in obj]

    # -- dataclasses / pydantic -------------------------------------------
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sanitise(dataclasses.asdict(obj))
    if hasattr(obj, "model_dump"):
        return sanitise(obj.model_dump())

    return obj


class SafeJSONResponse(JSONResponse):
    """Default response class for the API."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            sanitise(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
