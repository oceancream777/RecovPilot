from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

WEBHOOK_LOG_PATH = Path(
    os.environ.get("WEBHOOK_LOG_PATH", "data/webhooks_log.jsonl")
)


def flatten_json(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        field = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(
                flatten_json(value, prefix=field, separator=separator)
            )
        else:
            flattened[field] = value
    return flattened


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _append_json_line(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(record),
            default=_json_default,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("Webhook log append did not write the complete JSON record")
    finally:
        os.close(descriptor)


async def append_webhook_payload(
    raw_payload: Mapping[str, Any],
    *,
    normalized_fields: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    record = flatten_json(raw_payload)
    if normalized_fields:
        record.update(dict(normalized_fields))
    destination = Path(path) if path is not None else WEBHOOK_LOG_PATH
    await asyncio.to_thread(_append_json_line, destination, record)
    return record
