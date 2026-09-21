"""Dump TypeSafe SDK request/response JSON schemas to tests/fixtures/typesafe_schema.json.

The installed `typesafe-sdk` (0.7.1) does not expose its wire-level pydantic
models under a `typesafe_sdk.types` module as originally guessed. Inspecting
the package (`uv run python -c "import typesafe_sdk, pkgutil; ..."`) shows the
real models live in `typesafe_sdk._schemas.models`:

    SystemOneRequest   -- POST /v1/systemone request body
    SystemOneResponse  -- POST /v1/systemone response body

(`typesafe_sdk._core.question_types`/`.response_types` hold the SDK's own
ergonomic wrapper types used by `TypeSafeClient.system_one`, not the wire
schema; `_schemas.models` is the wire-level pydantic source of truth used
for validation, hence the choice here.)
"""

import json
import pathlib

import pydantic
from typesafe_sdk._schemas.models import SystemOneRequest, SystemOneResponse

out = {
    "request": pydantic.TypeAdapter(SystemOneRequest).json_schema(),
    "response": pydantic.TypeAdapter(SystemOneResponse).json_schema(),
}
pathlib.Path("tests/fixtures/typesafe_schema.json").write_text(
    json.dumps(out, indent=2, sort_keys=True)
)
