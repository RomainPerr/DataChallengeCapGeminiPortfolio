"""Small, readable file-backed cache wrapper for function calls."""

from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import asdict, fields, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

CACHE_DIR = Path(__file__).resolve().parent / "cache"
MAX_INLINE_ARGUMENTS = 240


def _cache_argument_key(bound: inspect.BoundArguments) -> str:
    """Keep short keys readable and hash large argument representations."""
    arguments = ",".join(
        f"{name}={value!r}" for name, value in bound.arguments.items()
    )
    if len(arguments) <= MAX_INLINE_ARGUMENTS:
        return arguments
    digest = hashlib.sha256(arguments.encode("utf-8")).hexdigest()
    return f"args_sha256={digest}"


def _cache_stem(function_name: str, argument_key: str) -> Path:
    return CACHE_DIR / f"{function_name}({argument_key})"


def _with_extension(stem: Path, extension: str) -> Path:
    """Append an extension without interpreting dots in readable keys."""
    return Path(f"{stem}{extension}")


def _resolve_qualified_name(module_name: str, qualified_name: str) -> type[Any]:
    value: Any = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        value = getattr(value, part)
    return value


def _restore_value(annotation: Any, value: Any) -> Any:
    """Restore the small amount of structure used by cached dataclasses."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        item_type = args[0] if args else Any
        return tuple(_restore_value(item_type, item) for item in value)
    if origin is list:
        item_type = args[0] if args else Any
        return [_restore_value(item_type, item) for item in value]
    if origin is dict:
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _restore_value(key_type, key): _restore_value(value_type, item)
            for key, item in value.items()
        }
    if isinstance(annotation, type) and is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return annotation(
            **{
                field.name: _restore_value(
                    hints.get(field.name, field.type), value[field.name]
                )
                for field in fields(annotation)
            }
        )
    return value


def _read_cache(stem: Path, returns_dataframe: bool) -> Any | None:
    csv_path = _with_extension(stem, ".csv")
    if csv_path.is_file():
        if returns_dataframe:
            import pandas as pd

            return pd.read_csv(csv_path)
        with csv_path.open(newline="", encoding="utf-8") as cache_file:
            row = next(csv.reader(cache_file))
        return json.loads(row[0])

    npy_path = _with_extension(stem, ".npy")
    if npy_path.is_file():
        import numpy as np

        return np.load(npy_path, allow_pickle=False)

    json_path = _with_extension(stem, ".json")
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("kind") != "dataclass":
            return payload["value"]
        result_type = _resolve_qualified_name(
            payload["module"], payload["qualname"]
        )
        return _restore_value(result_type, payload["value"])

    return None


def _write_cache(stem: Path, result: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if is_dataclass(result) and not isinstance(result, type):
        payload = {
            "kind": "dataclass",
            "module": result.__class__.__module__,
            "qualname": result.__class__.__qualname__,
            "value": asdict(result),
        }
        _with_extension(stem, ".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    if hasattr(result, "to_csv"):
        result.to_csv(_with_extension(stem, ".csv"), index=False)
        return

    try:
        import numpy as np

        if isinstance(result, np.ndarray):
            np.save(_with_extension(stem, ".npy"), result)
            return
    except ImportError:
        pass

    with _with_extension(stem, ".csv").open(
        "w", newline="", encoding="utf-8"
    ) as cache_file:
        csv.writer(cache_file).writerow([json.dumps(result, ensure_ascii=False)])


def get_cache_version_of[T: Callable[..., Any]](
    f: T,
    *,
    cache_if: Callable[[inspect.BoundArguments], bool] | None = None,
    cache_key: Callable[[inspect.BoundArguments], str] | None = None,
) -> T:
    """Cache a function result using readable files and bounded filenames.

    Short calls retain the original readable ``function(args).csv`` format.
    Large calls use a SHA-256 argument key so DataFrames, prompts, and model
    objects do not produce unwieldy filenames. ``cache_if`` allows tests or
    callers with injected clients to opt out of caching. ``cache_key`` allows
    callers to hash a stable semantic payload instead of object reprs.
    """
    signature = inspect.signature(f)
    return_annotation = signature.return_annotation
    returns_dataframe = (
        getattr(return_annotation, "__name__", None) == "DataFrame"
        or return_annotation in {"pd.DataFrame", "pandas.DataFrame"}
    )

    @wraps(f)
    def cache_version_of_f(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if cache_if is not None and not cache_if(bound):
            return f(*args, **kwargs)

        argument_key = (
            cache_key(bound)
            if cache_key is not None
            else _cache_argument_key(bound)
        )
        stem = _cache_stem(f.__name__, argument_key)
        cached = _read_cache(stem, returns_dataframe)
        if cached is not None:
            return cached

        result = f(*args, **kwargs)
        _write_cache(stem, result)
        return result

    return cache_version_of_f  # type: ignore[return-value]
