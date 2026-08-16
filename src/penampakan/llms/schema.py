"""Compile the provider-neutral action schema into named provider subsets.

The library's action schema is a root discriminated union. Provider structured
output APIs accept different, evolving JSON Schema subsets, so the schema is
never passed through unchanged. Every adapter compiles it into a named,
versioned, fingerprinted provider subset here, and every weakening is either
rejected or recorded in the one keyword table below.

Provider references:

- OpenAI Structured Outputs
  <https://developers.openai.com/api/docs/guides/structured-outputs>
- Anthropic Structured Outputs
  <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- Anthropic strict tool use
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use>
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING, Final, cast

from penampakan.errors import ConfigurationError
from penampakan.models import JsonValue

if TYPE_CHECKING:
    JsonObject = dict[str, JsonValue]
else:
    JsonObject = dict

SCHEMA_COMPILER_VERSION: Final = "provider-schema-v1"

# The single property that carries the action in every compiled envelope.
ENVELOPE_PROPERTY: Final = "action"

_REF_PREFIX: Final = "#/$defs/"
_DEF_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SchemaTarget(str, Enum):
    """A named provider schema subset."""

    OPENAI_STRICT = "openai_strict"
    ANTHROPIC_STRICT = "anthropic_strict"
    JSON_ONLY = "json_only"


class _Disposition(str, Enum):
    """What the compiler does with one JSON Schema keyword."""

    PRESERVE = "preserve"
    # Dropped from the provider schema; still enforced by local post-validation
    # against the original provider-neutral schema.
    DROP = "drop"
    REJECT = "reject"


# The one explicit keyword table. A keyword absent from this table is rejected,
# so an unrecognized construct can never be silently weakened.
_KEYWORD_TABLE: Final[dict[str, dict[SchemaTarget, _Disposition]]] = {
    keyword: {
        SchemaTarget.OPENAI_STRICT: openai,
        SchemaTarget.ANTHROPIC_STRICT: anthropic,
    }
    for keyword, openai, anthropic in (
        # Structural keywords the compiler itself rewrites.
        ("type", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("properties", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("required", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("additionalProperties", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("items", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("anyOf", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("oneOf", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("$ref", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("$defs", _Disposition.PRESERVE, _Disposition.PRESERVE),
        # Value constraints.
        ("const", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("enum", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("format", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("pattern", _Disposition.PRESERVE, _Disposition.DROP),
        ("minimum", _Disposition.PRESERVE, _Disposition.DROP),
        ("maximum", _Disposition.PRESERVE, _Disposition.DROP),
        ("exclusiveMinimum", _Disposition.PRESERVE, _Disposition.DROP),
        ("exclusiveMaximum", _Disposition.PRESERVE, _Disposition.DROP),
        ("multipleOf", _Disposition.PRESERVE, _Disposition.DROP),
        ("minItems", _Disposition.PRESERVE, _Disposition.DROP),
        ("maxItems", _Disposition.PRESERVE, _Disposition.DROP),
        # Neither documented subset accepts string length bounds.
        ("minLength", _Disposition.DROP, _Disposition.DROP),
        ("maxLength", _Disposition.DROP, _Disposition.DROP),
        # Annotations that carry no validation weight for the provider.
        ("description", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("title", _Disposition.PRESERVE, _Disposition.PRESERVE),
        ("default", _Disposition.DROP, _Disposition.DROP),
    )
}


@dataclass(frozen=True, slots=True)
class _ProviderLimits:
    """Documented limits for the supported provider API version."""

    api_version: str
    max_nesting_depth: int
    max_total_properties: int
    max_enum_values_total: int
    max_enum_string_budget_threshold: int
    max_enum_total_string_length: int
    max_schema_string_length: int


@dataclass(frozen=True, slots=True)
class _ProviderProfile:
    """One target's structural rules and documented limits."""

    target: SchemaTarget
    limits: _ProviderLimits
    # OpenAI strict output requires every property of every reachable object to
    # be required, with previously optional fields expressed as nullable unions.
    require_all_properties: bool


_OPENAI_PROFILE: Final = _ProviderProfile(
    target=SchemaTarget.OPENAI_STRICT,
    limits=_ProviderLimits(
        api_version="openai-responses-2026-05",
        max_nesting_depth=10,
        max_total_properties=5_000,
        max_enum_values_total=1_000,
        max_enum_string_budget_threshold=250,
        max_enum_total_string_length=15_000,
        max_schema_string_length=120_000,
    ),
    require_all_properties=True,
)

_ANTHROPIC_PROFILE: Final = _ProviderProfile(
    target=SchemaTarget.ANTHROPIC_STRICT,
    limits=_ProviderLimits(
        api_version="anthropic-messages-2023-06-01",
        max_nesting_depth=10,
        max_total_properties=5_000,
        max_enum_values_total=1_000,
        max_enum_string_budget_threshold=250,
        max_enum_total_string_length=15_000,
        max_schema_string_length=120_000,
    ),
    require_all_properties=False,
)

_PROFILES: Final[dict[SchemaTarget, _ProviderProfile]] = {
    SchemaTarget.OPENAI_STRICT: _OPENAI_PROFILE,
    SchemaTarget.ANTHROPIC_STRICT: _ANTHROPIC_PROFILE,
}


@dataclass(frozen=True, slots=True)
class CompiledSchema:
    """A canonicalized, fingerprinted provider schema subset."""

    target: SchemaTarget
    schema: dict[str, JsonValue]
    fingerprint_sha256: str
    compiler_version: str


def canonical_json(value: JsonValue) -> str:
    """Serialize a JSON value canonically for fingerprints and cache keys."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(target: SchemaTarget, schema: JsonValue) -> str:
    material = "\n".join((SCHEMA_COMPILER_VERSION, target.value, canonical_json(schema)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _fail(code: str, cause: Exception | None = None) -> ConfigurationError:
    # Schema text is caller data; only a stable code is ever exposed.
    return ConfigurationError(code=code, cause=cause)


def _as_object(value: object, code: str) -> JsonObject:
    if not isinstance(value, dict):
        raise _fail(code)
    for key in value:
        if not isinstance(key, str):
            raise _fail(code)
    return cast(JsonObject, value)


@dataclass(slots=True)
class _Node:
    """One compiled node paired with its inverse for round-trip verification."""

    compiled: JsonObject
    restored: JsonObject


@dataclass(slots=True)
class _Counters:
    total_properties: int = 0
    enum_values: int = 0
    enum_string_length: int = 0
    definition_depth: dict[str, int] = field(default_factory=dict)


class _Compiler:
    """Compile one provider-neutral schema into a single target subset."""

    def __init__(self, profile: _ProviderProfile, definitions: Mapping[str, JsonValue]) -> None:
        self._profile = profile
        self._definitions = definitions
        self._counters = _Counters()
        self._referenced: set[str] = set()

    @property
    def counters(self) -> _Counters:
        """Return the accumulated provider-limit counters."""
        return self._counters

    @property
    def referenced(self) -> set[str]:
        """Return the local definition names reached from the compiled schema."""
        return self._referenced

    def node(self, value: object) -> _Node:
        """Compile one schema node and build its restoration."""
        node = _as_object(value, "schema_node_invalid")
        compiled: JsonObject = {}
        restored: JsonObject = {}
        for keyword in node:
            disposition = self._disposition(keyword)
            if disposition is _Disposition.REJECT:
                raise _fail("unsupported_schema_keyword")
        for keyword, item in node.items():
            if self._disposition(keyword) is _Disposition.DROP:
                continue
            if keyword in {"properties", "required", "additionalProperties"}:
                continue
            if keyword == "items":
                child = self.node(item)
                compiled["items"] = child.compiled
                restored["items"] = child.restored
                continue
            if keyword in {"anyOf", "oneOf"}:
                self._union(keyword, item, compiled, restored)
                continue
            if keyword == "$ref":
                reference = self._reference(item)
                compiled["$ref"] = reference
                restored["$ref"] = reference
                continue
            if keyword == "$defs":
                raise _fail("nested_schema_definitions")
            if keyword == "enum":
                values = self._enum(item)
                compiled["enum"] = values
                restored["enum"] = values
                continue
            compiled[keyword] = item
            restored[keyword] = item
        if "properties" in node or node.get("type") == "object":
            self._object(node, compiled, restored)
        return _Node(compiled=compiled, restored=restored)

    def _disposition(self, keyword: str) -> _Disposition:
        row = _KEYWORD_TABLE.get(keyword)
        if row is None:
            return _Disposition.REJECT
        return row[self._profile.target]

    def _enum(self, value: object) -> JsonValue:
        if not isinstance(value, list) or not value:
            raise _fail("unsupported_schema_enum")
        self._counters.enum_values += len(value)
        for item in value:
            if isinstance(item, str):
                self._counters.enum_string_length += len(item)
            elif not isinstance(item, (int, float, bool)) and item is not None:
                raise _fail("unsupported_schema_enum")
        return cast(JsonValue, list(value))

    def _reference(self, value: object) -> str:
        if not isinstance(value, str) or not value.startswith(_REF_PREFIX):
            # Remote and non-local references cannot be verified locally.
            raise _fail("unsupported_schema_reference")
        name = value[len(_REF_PREFIX) :].replace("~1", "/").replace("~0", "~")
        if not _DEF_NAME.fullmatch(name) or name not in self._definitions:
            raise _fail("schema_reference_unresolved")
        self._referenced.add(name)
        return value

    def _union(
        self,
        keyword: str,
        value: object,
        compiled: JsonObject,
        restored: JsonObject,
    ) -> None:
        if not isinstance(value, list) or len(value) < 2:
            raise _fail("unsupported_schema_union")
        branches = [self.node(branch) for branch in value]
        if keyword == "oneOf":
            # A nested oneOf is only lowered when its branches are provably
            # distinguishable, exactly like the root union.
            _require_mutually_exclusive([branch.compiled for branch in branches])
        compiled["anyOf"] = [branch.compiled for branch in branches]
        restored[keyword] = [branch.restored for branch in branches]

    def _object(self, node: JsonObject, compiled: JsonObject, restored: JsonObject) -> None:
        properties = _as_object(node.get("properties", {}), "schema_properties_invalid")
        original_required = _string_list(node.get("required", []), "schema_required_invalid")
        unknown = [name for name in original_required if name not in properties]
        if unknown:
            raise _fail("schema_required_unknown_property")
        additional = node.get("additionalProperties", False)
        if additional is not False:
            # Open objects cannot be validated strictly by either provider.
            raise _fail("unsupported_additional_properties")
        compiled_properties: JsonObject = {}
        restored_properties: JsonObject = {}
        for name, definition in properties.items():
            self._counters.total_properties += 1
            child = self.node(definition)
            optional = name not in original_required
            if optional and self._profile.require_all_properties:
                compiled_properties[name] = _nullable(child.compiled)
            else:
                compiled_properties[name] = child.compiled
            restored_properties[name] = child.restored
        compiled["type"] = "object"
        compiled["properties"] = compiled_properties
        compiled["required"] = (
            list(properties)
            if self._profile.require_all_properties
            else [name for name in properties if name in original_required]
        )
        compiled["additionalProperties"] = False
        restored["type"] = "object"
        restored["properties"] = restored_properties
        restored["required"] = list(original_required)
        restored["additionalProperties"] = False


def _string_list(value: object, code: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _fail(code)
    return list(cast("list[str]", value))


def _nullable(node: JsonObject) -> JsonObject:
    """Return a required nullable union for a previously optional property."""
    branches = node.get("anyOf")
    if isinstance(branches, list):
        if any(isinstance(item, dict) and item.get("type") == "null" for item in branches):
            return node
        return {"anyOf": [*branches, {"type": "null"}]}
    return {"anyOf": [cast(JsonValue, node), {"type": "null"}]}


def _const_map(branch: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    properties = branch.get("properties")
    required = branch.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return {}
    result: dict[str, JsonValue] = {}
    for name, definition in properties.items():
        if name not in required or not isinstance(definition, dict):
            continue
        if "const" in definition:
            result[name] = definition["const"]
    return result


def _branch_properties(branch: Mapping[str, JsonValue]) -> frozenset[str]:
    properties = branch.get("properties")
    if not isinstance(properties, dict):
        return frozenset()
    return frozenset(properties)


def _exclusive(left: Mapping[str, JsonValue], right: Mapping[str, JsonValue]) -> bool:
    left_consts = _const_map(left)
    right_consts = _const_map(right)
    for name, value in left_consts.items():
        if name in right_consts and right_consts[name] != value:
            return True
    # A required const property that the other branch forbids also separates the
    # branches, because every compiled object closes additionalProperties.
    if any(name not in _branch_properties(right) for name in left_consts):
        return True
    return any(name not in _branch_properties(left) for name in right_consts)


def _require_mutually_exclusive(branches: Sequence[Mapping[str, JsonValue]]) -> None:
    signatures: set[str] = set()
    for branch in branches:
        consts = _const_map(branch)
        if not consts:
            raise _fail("unsupported_schema_union")
        signature = canonical_json(cast(JsonValue, dict(sorted(consts.items()))))
        if signature in signatures:
            raise _fail("schema_discriminator_conflict")
        signatures.add(signature)
    for index, left in enumerate(branches):
        for right in branches[index + 1 :]:
            if not _exclusive(left, right):
                raise _fail("schema_discriminator_conflict")


def _root_branches(schema: Mapping[str, JsonValue]) -> tuple[str, list[JsonValue]]:
    for keyword in ("oneOf", "anyOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list) and branches:
            return keyword, list(branches)
    raise _fail("schema_root_invalid")


def _envelope(action: JsonValue, definitions: Mapping[str, JsonValue]) -> JsonObject:
    envelope: JsonObject = {
        "type": "object",
        "properties": {ENVELOPE_PROPERTY: action},
        "required": [ENVELOPE_PROPERTY],
        "additionalProperties": False,
    }
    envelope["$defs"] = dict(definitions)
    return envelope


def _depth(node: JsonValue, definitions: Mapping[str, JsonValue], seen: frozenset[str]) -> int:
    if not isinstance(node, dict):
        return 0
    reference = node.get("$ref")
    if isinstance(reference, str) and reference.startswith(_REF_PREFIX):
        name = reference[len(_REF_PREFIX) :]
        if name in seen:
            raise _fail("unsupported_recursive_schema")
        target = definitions.get(name)
        if target is None:
            raise _fail("schema_reference_unresolved")
        return _depth(target, definitions, seen | {name})
    depth = 0
    properties = node.get("properties")
    if isinstance(properties, dict):
        for child in properties.values():
            depth = max(depth, 1 + _depth(child, definitions, seen))
    items = node.get("items")
    if items is not None:
        depth = max(depth, 1 + _depth(items, definitions, seen))
    for keyword in ("anyOf", "oneOf"):
        branches = node.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                depth = max(depth, _depth(branch, definitions, seen))
    return depth


def _check_limits(profile: _ProviderProfile, schema: JsonObject, counters: _Counters) -> None:
    limits = profile.limits
    definitions = _as_object(schema.get("$defs", {}), "schema_definitions_invalid")
    if _depth(cast(JsonValue, schema), definitions, frozenset()) > limits.max_nesting_depth:
        raise _fail("schema_depth_limit_exceeded")
    if counters.total_properties > limits.max_total_properties:
        raise _fail("schema_property_limit_exceeded")
    if counters.enum_values > limits.max_enum_values_total:
        raise _fail("schema_enum_limit_exceeded")
    if (
        counters.enum_values > limits.max_enum_string_budget_threshold
        and counters.enum_string_length > limits.max_enum_total_string_length
    ):
        raise _fail("schema_enum_limit_exceeded")
    if len(canonical_json(cast(JsonValue, schema))) > limits.max_schema_string_length:
        raise _fail("schema_size_limit_exceeded")


def _strip_dropped(value: JsonValue, target: SchemaTarget) -> JsonValue:
    """Remove keywords this target drops, for round-trip comparison."""
    if isinstance(value, list):
        return [_strip_dropped(item, target) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, JsonValue] = {}
    for keyword, item in value.items():
        row = _KEYWORD_TABLE.get(keyword)
        if row is not None and row[target] is _Disposition.DROP:
            continue
        result[keyword] = _strip_dropped(item, target)
    if value.get("type") == "object" or "properties" in value:
        result.setdefault("type", "object")
        result.setdefault("properties", {})
        result.setdefault("required", [])
        result["additionalProperties"] = False
    return result


def _compile_strict(schema: Mapping[str, JsonValue], profile: _ProviderProfile) -> JsonObject:
    definitions = _as_object(schema.get("$defs", {}), "schema_definitions_invalid")
    keyword, branches = _root_branches(schema)
    for extra in schema:
        if extra not in {"$defs", "oneOf", "anyOf", "title", "description"}:
            raise _fail("schema_root_invalid")
    compiler = _Compiler(profile, definitions)
    compiled_definitions: JsonObject = {}
    restored_definitions: JsonObject = {}
    for name, definition in definitions.items():
        if not _DEF_NAME.fullmatch(name):
            raise _fail("unsupported_schema_definition_name")
        node = compiler.node(definition)
        compiled_definitions[name] = node.compiled
        restored_definitions[name] = node.restored
    compiled_branches = [compiler.node(branch) for branch in branches]
    _require_mutually_exclusive([branch.compiled for branch in compiled_branches])
    action: JsonObject = {"anyOf": [branch.compiled for branch in compiled_branches]}
    compiled = _envelope(action, compiled_definitions)
    if "description" in schema:
        compiled["description"] = schema["description"]
    _check_limits(profile, compiled, compiler.counters)
    unresolved = compiler.referenced - set(compiled_definitions)
    if unresolved:
        raise _fail("schema_reference_unresolved")
    restored: JsonObject = {keyword: [branch.restored for branch in compiled_branches]}
    if "description" in schema:
        restored["description"] = schema["description"]
    if restored_definitions:
        restored["$defs"] = restored_definitions
    expected = _strip_dropped(cast(JsonValue, dict(schema)), profile.target)
    expected = _without_annotations(expected)
    if canonical_json(_without_annotations(cast(JsonValue, restored))) != canonical_json(expected):
        raise _fail("schema_roundtrip_failed")
    return compiled


def _without_annotations(value: JsonValue) -> JsonValue:
    """Drop root-only annotations that never reach a provider schema."""
    if isinstance(value, dict):
        return {
            key: _without_annotations(item) for key, item in value.items() if key not in {"title"}
        }
    if isinstance(value, list):
        return [_without_annotations(item) for item in value]
    return value


def _compile_json_only(schema: Mapping[str, JsonValue]) -> JsonObject:
    """Wrap the original schema for prompt use without any weakening claim."""
    definitions = _as_object(schema.get("$defs", {}), "schema_definitions_invalid")
    keyword, branches = _root_branches(schema)
    action: JsonObject = {keyword: list(branches)}
    return _envelope(action, definitions)


def compile_action_schema(
    schema: Mapping[str, JsonValue],
    *,
    target: SchemaTarget,
) -> CompiledSchema:
    """Compile the provider-neutral action schema for one provider target.

    The result is canonical and fingerprinted, so the compiler version and
    fingerprint can feed trace metadata and benchmark response-cache keys.
    """
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping")
    if not isinstance(target, SchemaTarget):
        raise TypeError("target must be a SchemaTarget")
    if target is SchemaTarget.JSON_ONLY:
        compiled = _compile_json_only(schema)
    else:
        compiled = _compile_strict(schema, _PROFILES[target])
    canonical = cast(JsonObject, json.loads(canonical_json(cast(JsonValue, compiled))))
    return CompiledSchema(
        target=target,
        schema=canonical,
        fingerprint_sha256=_fingerprint(target, cast(JsonValue, canonical)),
        compiler_version=SCHEMA_COMPILER_VERSION,
    )


def unwrap_action_envelope(value: object) -> dict[str, JsonValue]:
    """Return the action object carried by a compiled provider envelope."""
    if not isinstance(value, dict):
        raise ValueError("provider output must be one JSON object")
    keys = set(value)
    if keys != {ENVELOPE_PROPERTY}:
        raise ValueError("provider output must carry exactly one action property")
    action = value[ENVELOPE_PROPERTY]
    if not isinstance(action, dict) or any(not isinstance(key, str) for key in action):
        raise ValueError("action must be one JSON object")
    return cast(JsonObject, action)


def _resolve(node: JsonValue, definitions: Mapping[str, JsonValue]) -> JsonValue:
    seen: set[str] = set()
    while isinstance(node, dict):
        reference = node.get("$ref")
        if not isinstance(reference, str) or not reference.startswith(_REF_PREFIX):
            return node
        name = reference[len(_REF_PREFIX) :]
        if name in seen:
            raise ValueError("recursive schema reference")
        seen.add(name)
        resolved = definitions.get(name)
        if resolved is None:
            raise ValueError("unresolved schema reference")
        node = resolved
    return node


def _discriminated_branch(
    value: object,
    branches: Sequence[JsonValue],
    definitions: Mapping[str, JsonValue],
) -> JsonObject | None:
    """Select the one union branch whose required discriminators match.

    Selection uses the same const discriminators the compiler proved unique, so
    it never depends on constraints the provider was allowed to drop.
    """
    candidates: list[JsonObject] = []
    for branch in branches:
        resolved = _resolve(branch, definitions)
        if not isinstance(resolved, dict):
            continue
        consts = _const_map(resolved)
        if isinstance(value, dict) and consts:
            if all(value.get(name) == expected for name, expected in consts.items()):
                candidates.append(resolved)
            continue
        if not consts and _type_matches(value, str(resolved.get("type", ""))):
            candidates.append(resolved)
    if len(candidates) == 1:
        return candidates[0]
    return None


def prune_optional_nulls(
    value: JsonValue,
    schema: Mapping[str, JsonValue],
    *,
    definitions: Mapping[str, JsonValue] | None = None,
) -> JsonValue:
    """Drop null values that a strict target introduced for optional fields.

    A target that requires every property expresses a previously optional field
    as a required nullable union. Removing those explicit nulls restores the
    provider-neutral shape before post-validation.
    """
    resolved_definitions = (
        _as_object(schema.get("$defs", {}), "schema_definitions_invalid")
        if definitions is None
        else definitions
    )
    node = _resolve(cast(JsonValue, dict(schema)), resolved_definitions)
    if not isinstance(node, dict):
        return value
    for keyword in ("oneOf", "anyOf"):
        branches = node.get(keyword)
        if isinstance(branches, list):
            branch = _discriminated_branch(value, branches, resolved_definitions)
            if branch is None:
                return value
            return prune_optional_nulls(value, branch, definitions=resolved_definitions)
    if isinstance(value, dict):
        properties = node.get("properties")
        required = node.get("required")
        required_names = set(required) if isinstance(required, list) else set()
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            child = properties.get(key) if isinstance(properties, dict) else None
            if item is None and key not in required_names:
                continue
            if isinstance(child, dict):
                result[key] = prune_optional_nulls(
                    item,
                    cast(Mapping[str, JsonValue], child),
                    definitions=resolved_definitions,
                )
            else:
                result[key] = item
        return result
    if isinstance(value, list):
        items = node.get("items")
        if isinstance(items, dict):
            return [
                prune_optional_nulls(
                    item,
                    cast(Mapping[str, JsonValue], items),
                    definitions=resolved_definitions,
                )
                for item in value
            ]
    return value


def _type_matches(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def validate_action_instance(
    value: object,
    schema: Mapping[str, JsonValue],
    *,
    definitions: Mapping[str, JsonValue] | None = None,
    path: str = "$",
) -> tuple[str, ...]:
    """Validate a parsed instance against the original action schema subset.

    Returns redacted ``location:keyword`` findings. Only schema locations and
    keyword names are reported, never instance content.
    """
    resolved_definitions = (
        _as_object(schema.get("$defs", {}), "schema_definitions_invalid")
        if definitions is None
        else definitions
    )
    try:
        node = _resolve(cast(JsonValue, dict(schema)), resolved_definitions)
    except ValueError:
        return (f"{path}:$ref",)
    if not isinstance(node, dict):
        return (f"{path}:schema",)
    errors: list[str] = []
    for keyword in ("oneOf", "anyOf"):
        branches = node.get(keyword)
        if not isinstance(branches, list):
            continue
        matches = [
            branch
            for branch in branches
            if isinstance(_resolve(branch, resolved_definitions), dict)
            and not validate_action_instance(
                value,
                cast(Mapping[str, JsonValue], _resolve(branch, resolved_definitions)),
                definitions=resolved_definitions,
                path=path,
            )
        ]
        if not matches or (keyword == "oneOf" and len(matches) != 1):
            errors.append(f"{path}:{keyword}")
        return tuple(errors)
    expected = node.get("type")
    if isinstance(expected, str) and not _type_matches(value, expected):
        return (f"{path}:type",)
    if isinstance(expected, list) and not any(
        isinstance(item, str) and _type_matches(value, item) for item in expected
    ):
        return (f"{path}:type",)
    if "const" in node and value != node["const"]:
        errors.append(f"{path}:const")
    enum_values = node.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}:enum")
    if isinstance(value, str):
        errors.extend(_string_errors(value, node, path))
    number = _number(value)
    if number is not None:
        errors.extend(_number_errors(number, node, path))
    if isinstance(value, dict):
        errors.extend(_object_errors(value, node, resolved_definitions, path))
    if isinstance(value, list):
        errors.extend(_array_errors(value, node, resolved_definitions, path))
    return tuple(errors)


def _string_errors(value: str, node: Mapping[str, JsonValue], path: str) -> list[str]:
    errors: list[str] = []
    minimum = node.get("minLength")
    maximum = node.get("maxLength")
    pattern = node.get("pattern")
    if isinstance(minimum, int) and len(value) < minimum:
        errors.append(f"{path}:minLength")
    if isinstance(maximum, int) and len(value) > maximum:
        errors.append(f"{path}:maxLength")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        errors.append(f"{path}:pattern")
    return errors


def _number_errors(value: float, node: Mapping[str, JsonValue], path: str) -> list[str]:
    errors: list[str] = []
    if not math.isfinite(value):
        return [f"{path}:type"]
    bounds: tuple[tuple[str, Callable[[float], bool]], ...] = (
        ("minimum", lambda limit: value < limit),
        ("maximum", lambda limit: value > limit),
        ("exclusiveMinimum", lambda limit: value <= limit),
        ("exclusiveMaximum", lambda limit: value >= limit),
    )
    for keyword, violates in bounds:
        limit = _number(node.get(keyword))
        if limit is not None and violates(limit):
            errors.append(f"{path}:{keyword}")
    multiple = _number(node.get("multipleOf"))
    if multiple is not None and multiple > 0.0 and not _is_exact_decimal_multiple(value, multiple):
        errors.append(f"{path}:multipleOf")
    return errors


def _is_exact_decimal_multiple(value: float, multiple: float) -> bool:
    """Compare JSON-number decimal values without binary floating-point modulo."""
    if not math.isfinite(multiple):
        return False
    try:
        quotient = Decimal(str(value)) / Decimal(str(multiple))
    except (InvalidOperation, ZeroDivisionError):
        return False
    return quotient == quotient.to_integral_value()


def _object_errors(
    value: Mapping[str, object],
    node: Mapping[str, JsonValue],
    definitions: Mapping[str, JsonValue],
    path: str,
) -> list[str]:
    errors: list[str] = []
    properties = node.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = node.get("required")
    if isinstance(required, list):
        errors.extend(
            f"{path}.{name}:required"
            for name in required
            if isinstance(name, str) and name not in value
        )
    if node.get("additionalProperties") is False:
        errors.extend(
            f"{path}.{_safe_key(key)}:additionalProperties"
            for key in value
            if key not in properties
        )
    for key, item in value.items():
        child = properties.get(key)
        if isinstance(child, dict):
            errors.extend(
                validate_action_instance(
                    item,
                    cast(Mapping[str, JsonValue], child),
                    definitions=definitions,
                    path=f"{path}.{_safe_key(key)}",
                )
            )
    return errors


def _array_errors(
    value: Sequence[object],
    node: Mapping[str, JsonValue],
    definitions: Mapping[str, JsonValue],
    path: str,
) -> list[str]:
    errors: list[str] = []
    minimum = node.get("minItems")
    maximum = node.get("maxItems")
    if isinstance(minimum, int) and len(value) < minimum:
        errors.append(f"{path}:minItems")
    if isinstance(maximum, int) and len(value) > maximum:
        errors.append(f"{path}:maxItems")
    items = node.get("items")
    if isinstance(items, dict):
        for index, item in enumerate(value):
            errors.extend(
                validate_action_instance(
                    item,
                    cast(Mapping[str, JsonValue], items),
                    definitions=definitions,
                    path=f"{path}[{index}]",
                )
            )
    return errors


_SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _safe_key(key: object) -> str:
    if isinstance(key, str) and _SAFE_KEY.fullmatch(key):
        return key
    return "[unknown]"


__all__ = [
    "ENVELOPE_PROPERTY",
    "SCHEMA_COMPILER_VERSION",
    "CompiledSchema",
    "SchemaTarget",
    "canonical_json",
    "compile_action_schema",
    "prune_optional_nulls",
    "unwrap_action_envelope",
    "validate_action_instance",
]
