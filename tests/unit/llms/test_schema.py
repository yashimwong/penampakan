"""Unit tests for the provider schema compiler."""

from __future__ import annotations

import json
from typing import cast

import pytest

from penampakan.errors import ConfigurationError
from penampakan.llms.schema import (
    ENVELOPE_PROPERTY,
    SCHEMA_COMPILER_VERSION,
    CompiledSchema,
    SchemaTarget,
    canonical_json,
    compile_action_schema,
    prune_optional_nulls,
    unwrap_action_envelope,
    validate_action_instance,
)
from penampakan.models import Capability, JsonValue, ToolSpec
from penampakan.perception.registry import ToolRegistry
from penampakan.reasoning.prompts import build_action_schema
from penampakan.tools.builtin import register_transform_tools
from penampakan.tools.vision import register_vision_tools

_STRICT_TARGETS = (SchemaTarget.OPENAI_STRICT, SchemaTarget.ANTHROPIC_STRICT)


def _library_schema() -> dict[str, JsonValue]:
    registry = ToolRegistry()
    register_vision_tools(registry, set(Capability))
    register_transform_tools(registry)
    return build_action_schema(registry.specs)


def _tool_spec(name: str = "probe") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Probe one declared value.",
        arguments_json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"asset_id": {"type": "string", "pattern": r"^img_[0-9a-f]{16,64}$"}},
            "required": ["asset_id"],
        },
    )


def _schema(**overrides: JsonValue) -> dict[str, JsonValue]:
    schema = build_action_schema((_tool_spec(),))
    schema.update(overrides)
    return schema


def _branches(compiled: CompiledSchema) -> list[dict[str, JsonValue]]:
    properties = cast(dict[str, JsonValue], compiled.schema["properties"])
    action = cast(dict[str, JsonValue], properties[ENVELOPE_PROPERTY])
    return [cast(dict[str, JsonValue], branch) for branch in cast(list[JsonValue], action["anyOf"])]


@pytest.mark.parametrize("target", list(SchemaTarget))
def test_compiles_the_library_schema_for_every_target(target: SchemaTarget) -> None:
    compiled = compile_action_schema(_library_schema(), target=target)
    assert compiled.target is target
    assert compiled.compiler_version == SCHEMA_COMPILER_VERSION
    assert len(compiled.fingerprint_sha256) == 64
    assert compiled.schema["type"] == "object"
    assert compiled.schema["required"] == [ENVELOPE_PROPERTY]
    assert compiled.schema["additionalProperties"] is False
    assert set(cast(dict[str, JsonValue], compiled.schema["properties"])) == {ENVELOPE_PROPERTY}


@pytest.mark.parametrize("target", list(SchemaTarget))
def test_compilation_is_idempotent_and_canonically_fingerprinted(target: SchemaTarget) -> None:
    schema = _library_schema()
    first = compile_action_schema(schema, target=target)
    second = compile_action_schema(dict(schema), target=target)
    assert first.schema == second.schema
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    # The compiled schema is already canonical, so recompiling it is stable.
    assert canonical_json(cast(JsonValue, first.schema)) == json.dumps(
        first.schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def test_fingerprints_differ_by_target_and_content() -> None:
    schema = _library_schema()
    fingerprints = {
        compile_action_schema(schema, target=target).fingerprint_sha256 for target in SchemaTarget
    }
    assert len(fingerprints) == len(SchemaTarget)
    other = compile_action_schema(
        build_action_schema((_tool_spec("other"),)), target=SchemaTarget.OPENAI_STRICT
    )
    baseline = compile_action_schema(
        build_action_schema((_tool_spec(),)), target=SchemaTarget.OPENAI_STRICT
    )
    assert other.fingerprint_sha256 != baseline.fingerprint_sha256


def test_root_union_is_lowered_to_a_nested_any_of() -> None:
    compiled = compile_action_schema(_schema(), target=SchemaTarget.OPENAI_STRICT)
    action = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], compiled.schema["properties"])[ENVELOPE_PROPERTY],
    )
    assert "oneOf" not in action
    assert isinstance(action["anyOf"], list)
    assert len(action["anyOf"]) == 2


def test_openai_target_requires_every_property_with_nullable_optionals() -> None:
    compiled = compile_action_schema(_schema(), target=SchemaTarget.OPENAI_STRICT)
    answer = next(
        branch
        for branch in _branches(compiled)
        if cast(dict[str, JsonValue], branch["properties"]).get("type") == {"const": "answer"}
    )
    properties = cast(dict[str, JsonValue], answer["properties"])
    assert set(cast(list[str], answer["required"])) == set(properties)
    assert answer["additionalProperties"] is False
    evidence = cast(dict[str, JsonValue], properties["evidence"])
    assert {"type": "null"} in cast(list[JsonValue], evidence["anyOf"])


def test_anthropic_target_keeps_optional_properties_optional() -> None:
    compiled = compile_action_schema(_schema(), target=SchemaTarget.ANTHROPIC_STRICT)
    answer = next(
        branch
        for branch in _branches(compiled)
        if cast(dict[str, JsonValue], branch["properties"]).get("type") == {"const": "answer"}
    )
    properties = cast(dict[str, JsonValue], answer["properties"])
    required = set(cast(list[str], answer["required"]))
    assert required == {"type", "status", "answer"}
    assert "evidence" in properties
    assert answer["additionalProperties"] is False


def test_anthropic_target_drops_constraints_outside_its_subset() -> None:
    compiled = compile_action_schema(_schema(), target=SchemaTarget.ANTHROPIC_STRICT)
    text = canonical_json(cast(JsonValue, compiled.schema))
    for keyword in ("pattern", "minimum", "maximum", "minItems", "maxItems"):
        assert keyword not in text
    openai = compile_action_schema(_schema(), target=SchemaTarget.OPENAI_STRICT)
    assert "pattern" in canonical_json(cast(JsonValue, openai.schema))


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_string_length_bounds_are_dropped_for_both_strict_targets(target: SchemaTarget) -> None:
    compiled = compile_action_schema(_schema(), target=target)
    text = canonical_json(cast(JsonValue, compiled.schema))
    assert "minLength" not in text
    assert "maxLength" not in text
    assert "default" not in text


def test_json_only_target_keeps_the_original_union_and_constraints() -> None:
    compiled = compile_action_schema(_schema(), target=SchemaTarget.JSON_ONLY)
    action = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], compiled.schema["properties"])[ENVELOPE_PROPERTY],
    )
    assert "oneOf" in action
    text = canonical_json(cast(JsonValue, compiled.schema))
    assert "maxLength" in text


def test_local_references_are_resolved_and_preserved() -> None:
    registry = ToolRegistry()
    register_transform_tools(registry)
    compiled = compile_action_schema(
        build_action_schema(registry.specs), target=SchemaTarget.OPENAI_STRICT
    )
    definitions = cast(dict[str, JsonValue], compiled.schema["$defs"])
    assert definitions
    text = canonical_json(cast(JsonValue, compiled.schema))
    for name in definitions:
        assert f"#/$defs/{name}" in text


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_unresolved_reference_fails_compilation(target: SchemaTarget) -> None:
    schema = _schema()
    branches = cast(list[JsonValue], schema["oneOf"])
    branch = cast(dict[str, JsonValue], branches[0])
    properties = cast(dict[str, JsonValue], branch["properties"])
    properties["arguments"] = {"$ref": "#/$defs/Absent"}
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(schema, target=target)
    assert failure.value.code == "schema_reference_unresolved"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_remote_reference_is_rejected(target: SchemaTarget) -> None:
    schema = _schema()
    branch = cast(dict[str, JsonValue], cast(list[JsonValue], schema["oneOf"])[0])
    cast(dict[str, JsonValue], branch["properties"])["arguments"] = {
        "$ref": "https://example.invalid/schema.json"
    }
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(schema, target=target)
    assert failure.value.code == "unsupported_schema_reference"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_recursive_reference_is_rejected(target: SchemaTarget) -> None:
    schema: dict[str, JsonValue] = {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"type": {"const": "a"}, "next": {"$ref": "#/$defs/Node"}},
                "required": ["type", "next"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"type": {"const": "b"}},
                "required": ["type"],
            },
        ],
        "$defs": {
            "Node": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"child": {"$ref": "#/$defs/Node"}},
                "required": ["child"],
            }
        },
    }
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(schema, target=target)
    assert failure.value.code == "unsupported_recursive_schema"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_union_without_a_discriminator_is_rejected(target: SchemaTarget) -> None:
    schema: dict[str, JsonValue] = {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        ]
    }
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(schema, target=target)
    assert failure.value.code == "unsupported_schema_union"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_duplicate_discriminator_is_rejected(target: SchemaTarget) -> None:
    branch: dict[str, JsonValue] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"type": {"const": "same"}, "value": {"type": "string"}},
        "required": ["type", "value"],
    }
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema({"oneOf": [branch, dict(branch)]}, target=target)
    assert failure.value.code == "schema_discriminator_conflict"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_unsupported_keyword_fails_rather_than_weakening(target: SchemaTarget) -> None:
    schema = _schema()
    branch = cast(dict[str, JsonValue], cast(list[JsonValue], schema["oneOf"])[0])
    cast(dict[str, JsonValue], branch["properties"])["purpose"] = {
        "type": "string",
        "not": {"const": "forbidden"},
    }
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(schema, target=target)
    assert failure.value.code == "unsupported_schema_keyword"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_open_object_is_rejected(target: SchemaTarget) -> None:
    schema = _schema()
    branch = cast(dict[str, JsonValue], cast(list[JsonValue], schema["oneOf"])[0])
    properties = cast(dict[str, JsonValue], branch["properties"])
    properties["arguments"] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {},
        "required": [],
    }
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(schema, target=target)
    assert failure.value.code == "unsupported_additional_properties"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_absent_additional_properties_is_closed_rather_than_rejected(
    target: SchemaTarget,
) -> None:
    schema = _schema()
    branch = cast(dict[str, JsonValue], cast(list[JsonValue], schema["oneOf"])[0])
    properties = cast(dict[str, JsonValue], branch["properties"])
    properties["arguments"] = {
        "type": "object",
        "properties": {"asset_id": {"type": "string"}},
        "required": ["asset_id"],
    }
    compiled = compile_action_schema(schema, target=target)
    tool_branch = next(
        item
        for item in _branches(compiled)
        if cast(dict[str, JsonValue], item["properties"]).get("type") == {"const": "tool"}
    )
    arguments = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], tool_branch["properties"])["arguments"],
    )
    assert arguments["additionalProperties"] is False


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_root_without_a_union_is_rejected(target: SchemaTarget) -> None:
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema({"type": "object"}, target=target)
    assert failure.value.code == "schema_root_invalid"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_depth_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    target: SchemaTarget,
) -> None:
    from penampakan.llms import schema as module

    profile = module._PROFILES[target]
    limits = module._ProviderLimits(
        api_version=profile.limits.api_version,
        max_nesting_depth=2,
        max_total_properties=profile.limits.max_total_properties,
        max_enum_values_total=profile.limits.max_enum_values_total,
        max_enum_string_budget_threshold=profile.limits.max_enum_string_budget_threshold,
        max_enum_total_string_length=profile.limits.max_enum_total_string_length,
        max_schema_string_length=profile.limits.max_schema_string_length,
    )
    monkeypatch.setitem(
        module._PROFILES,
        target,
        module._ProviderProfile(
            target=profile.target,
            limits=limits,
            require_all_properties=profile.require_all_properties,
        ),
    )
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(_library_schema(), target=target)
    assert failure.value.code == "schema_depth_limit_exceeded"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_property_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    target: SchemaTarget,
) -> None:
    from penampakan.llms import schema as module

    profile = module._PROFILES[target]
    limits = module._ProviderLimits(
        api_version=profile.limits.api_version,
        max_nesting_depth=profile.limits.max_nesting_depth,
        max_total_properties=1,
        max_enum_values_total=profile.limits.max_enum_values_total,
        max_enum_string_budget_threshold=profile.limits.max_enum_string_budget_threshold,
        max_enum_total_string_length=profile.limits.max_enum_total_string_length,
        max_schema_string_length=profile.limits.max_schema_string_length,
    )
    monkeypatch.setitem(
        module._PROFILES,
        target,
        module._ProviderProfile(
            target=profile.target,
            limits=limits,
            require_all_properties=profile.require_all_properties,
        ),
    )
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(_schema(), target=target)
    assert failure.value.code == "schema_property_limit_exceeded"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_enum_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    target: SchemaTarget,
) -> None:
    from penampakan.llms import schema as module

    profile = module._PROFILES[target]
    limits = module._ProviderLimits(
        api_version=profile.limits.api_version,
        max_nesting_depth=profile.limits.max_nesting_depth,
        max_total_properties=profile.limits.max_total_properties,
        max_enum_values_total=1,
        max_enum_string_budget_threshold=profile.limits.max_enum_string_budget_threshold,
        max_enum_total_string_length=profile.limits.max_enum_total_string_length,
        max_schema_string_length=profile.limits.max_schema_string_length,
    )
    monkeypatch.setitem(
        module._PROFILES,
        target,
        module._ProviderProfile(
            target=profile.target,
            limits=limits,
            require_all_properties=profile.require_all_properties,
        ),
    )
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(_schema(), target=target)
    assert failure.value.code == "schema_enum_limit_exceeded"


def test_round_trip_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from penampakan.llms import schema as module

    def _corrupt(value: JsonValue, target: SchemaTarget) -> JsonValue:
        return {"type": "object"}

    monkeypatch.setattr(module, "_strip_dropped", _corrupt)
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(_schema(), target=SchemaTarget.OPENAI_STRICT)
    assert failure.value.code == "schema_roundtrip_failed"


def test_compilation_errors_never_carry_schema_content() -> None:
    schema = _schema()
    branch = cast(dict[str, JsonValue], cast(list[JsonValue], schema["oneOf"])[0])
    cast(dict[str, JsonValue], branch["properties"])["purpose"] = {
        "type": "string",
        "not": {"const": "secret-marker-value"},
    }
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(schema, target=SchemaTarget.OPENAI_STRICT)
    rendered = f"{failure.value} {failure.value!r} {failure.value.args}"
    assert "secret-marker-value" not in rendered
    assert "purpose" not in rendered


def test_compile_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError):
        compile_action_schema(cast(dict[str, JsonValue], []), target=SchemaTarget.OPENAI_STRICT)
    with pytest.raises(TypeError):
        compile_action_schema(_schema(), target=cast(SchemaTarget, "openai_strict"))


def test_unwrap_action_envelope_requires_exactly_one_action() -> None:
    assert unwrap_action_envelope({"action": {"type": "answer"}}) == {"type": "answer"}
    for value in (
        [],
        {},
        {"type": "answer"},
        {"action": {"type": "answer"}, "extra": 1},
        {"action": "answer"},
    ):
        with pytest.raises(ValueError, match=r"\S"):
            unwrap_action_envelope(value)


def test_prune_optional_nulls_restores_the_neutral_shape() -> None:
    schema = _library_schema()
    envelope = {
        "action": {
            "type": "answer",
            "status": "answered",
            "answer": "The sign reads OPEN.",
            "evidence": None,
            "uncertainties": None,
        }
    }
    pruned = prune_optional_nulls(cast(JsonValue, unwrap_action_envelope(envelope)), schema)
    assert pruned == {"type": "answer", "status": "answered", "answer": "The sign reads OPEN."}
    assert validate_action_instance(pruned, schema) == ()


def test_prune_optional_nulls_keeps_required_and_nested_values() -> None:
    schema = _library_schema()
    action: JsonValue = {
        "type": "tool",
        "tool": "crop",
        "arguments": {
            "asset_id": "img_" + "a" * 16,
            "box": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9},
            "padding_fraction": None,
        },
        "purpose": "Inspect the label.",
    }
    pruned = cast(dict[str, JsonValue], prune_optional_nulls(action, schema))
    arguments = cast(dict[str, JsonValue], pruned["arguments"])
    assert "padding_fraction" not in arguments
    assert arguments["box"] == {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}
    assert validate_action_instance(pruned, schema) == ()


def test_post_validation_rejects_violating_instances() -> None:
    schema = _library_schema()
    for action in (
        {"type": "answer", "status": "unknown", "answer": "x"},
        {"type": "answer", "status": "answered", "answer": ""},
        {"type": "answer", "status": "answered", "answer": "x", "surplus": 1},
        {"type": "tool", "tool": "not_declared", "arguments": {}, "purpose": "x"},
        {"type": "tool", "tool": "crop", "arguments": {"asset_id": "nope"}, "purpose": "x"},
        {"type": "answer", "status": "answered", "answer": "x", "evidence": [{"supports": "x"}]},
        "not-an-object",
    ):
        assert validate_action_instance(action, schema), action


def test_post_validation_reports_only_safe_locations() -> None:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"purpose": {"type": "string", "maxLength": 3}},
        "required": ["purpose"],
    }
    findings = validate_action_instance({"purpose": "far-too-long-secret"}, schema)
    assert findings == ("$.purpose:maxLength",)
    surplus = validate_action_instance({"purpose": "abc", "leak me": "secret"}, schema)
    assert surplus == ("$.[unknown]:additionalProperties",)


def test_post_validation_checks_numeric_and_array_bounds() -> None:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 4, "multipleOf": 2},
            "items": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "string"}},
        },
        "required": ["count", "items"],
    }
    assert validate_action_instance({"count": 2, "items": ["a"]}, schema) == ()
    assert validate_action_instance({"count": 0, "items": ["a"]}, schema) == ("$.count:minimum",)
    assert validate_action_instance({"count": 3, "items": ["a"]}, schema) == ("$.count:multipleOf",)
    assert validate_action_instance({"count": 2, "items": []}, schema) == ("$.items:minItems",)
    assert validate_action_instance({"count": 2, "items": ["a", "b", "c"]}, schema) == (
        "$.items:maxItems",
    )
    assert validate_action_instance({"count": 2}, schema) == ("$.items:required",)


@pytest.mark.parametrize("value", [0.3, 0.5, -0.3])
def test_post_validation_accepts_fractional_decimal_multiples(value: float) -> None:
    schema: dict[str, JsonValue] = {"type": "number", "multipleOf": 0.1}

    assert validate_action_instance(value, schema) == ()


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_root_description_is_preserved_by_strict_compilation(target: SchemaTarget) -> None:
    compiled = compile_action_schema(
        _schema(description="Choose exactly one policy action."),
        target=target,
    )

    assert compiled.schema["description"] == "Choose exactly one policy action."


def _branch(discriminator: str, **properties: JsonValue) -> dict[str, JsonValue]:
    merged: dict[str, JsonValue] = {"type": {"const": discriminator}}
    merged.update(properties)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": merged,
        "required": ["type", *properties],
    }


def _union(*branches: JsonValue, **extra: JsonValue) -> dict[str, JsonValue]:
    schema: dict[str, JsonValue] = {"oneOf": list(branches)}
    schema.update(extra)
    return schema


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_malformed_schema_nodes_are_rejected_with_stable_codes(target: SchemaTarget) -> None:
    cases: tuple[tuple[JsonValue, str], ...] = (
        (
            _union(_branch("a", value={"type": "string"}), {"type": "object", "properties": []}),
            "schema_properties_invalid",
        ),
        (
            _union(
                _branch("a", value={"type": "string"}),
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"type": {"const": "b"}},
                    "required": [1],
                },
            ),
            "schema_required_invalid",
        ),
        (
            _union(
                _branch("a", value={"type": "string"}),
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"type": {"const": "b"}},
                    "required": ["type", "absent"],
                },
            ),
            "schema_required_unknown_property",
        ),
        (
            _union(_branch("a", value={"type": "string"}), _branch("b", value={"enum": []})),
            "unsupported_schema_enum",
        ),
        (
            _union(
                _branch("a", value={"type": "string"}),
                _branch("b", value={"enum": [{"nested": True}]}),
            ),
            "unsupported_schema_enum",
        ),
        (
            _union(
                _branch("a", value={"type": "string"}),
                _branch("b", value={"$defs": {"Inner": {"type": "string"}}}),
            ),
            "nested_schema_definitions",
        ),
        (
            _union(_branch("a", value={"type": "string"}), _branch("b", value={"anyOf": []})),
            "unsupported_schema_union",
        ),
        (_union("not-an-object", _branch("a")), "schema_node_invalid"),
        (
            _union(
                _branch("a", value={"type": "string"}),
                _branch("b"),
                **{"$defs": {"not a name": {"type": "string"}}},
            ),
            "unsupported_schema_definition_name",
        ),
        (
            _union(_branch("a", value={"type": "string"}), _branch("b"), **{"$defs": []}),
            "schema_definitions_invalid",
        ),
        (
            _union(_branch("a", value={"type": "string"}), _branch("b"), title="Action", extra=1),
            "schema_root_invalid",
        ),
    )
    for schema, code in cases:
        with pytest.raises(ConfigurationError) as failure:
            compile_action_schema(cast(dict[str, JsonValue], schema), target=target)
        assert failure.value.code == code, code


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_a_union_branch_that_forbids_a_discriminator_is_still_distinguishable(
    target: SchemaTarget,
) -> None:
    # The answer branch has no "tool" property at all, so a closed object keeps
    # it separable from every tool branch.
    compiled = compile_action_schema(
        _union(
            _branch("tool", tool={"const": "crop"}),
            _branch("answer", answer={"type": "string"}),
        ),
        target=target,
    )
    assert len(_branches(compiled)) == 2


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_nested_unions_are_lowered_and_verified(target: SchemaTarget) -> None:
    nested: JsonValue = {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"kind": {"const": "box"}},
                "required": ["kind"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"kind": {"const": "point"}},
                "required": ["kind"],
            },
        ]
    }
    compiled = compile_action_schema(
        _union(_branch("a", region=nested), _branch("b")), target=target
    )
    text = canonical_json(cast(JsonValue, compiled.schema))
    assert '"oneOf"' not in text


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_an_indistinguishable_nested_union_is_rejected(target: SchemaTarget) -> None:
    nested: JsonValue = {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"kind": {"type": "string"}},
                "required": ["kind"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"kind": {"type": "integer"}},
                "required": ["kind"],
            },
        ]
    }
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(_union(_branch("a", region=nested), _branch("b")), target=target)
    assert failure.value.code == "unsupported_schema_union"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_schema_size_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    target: SchemaTarget,
) -> None:
    from penampakan.llms import schema as module

    profile = module._PROFILES[target]
    limits = module._ProviderLimits(
        api_version=profile.limits.api_version,
        max_nesting_depth=profile.limits.max_nesting_depth,
        max_total_properties=profile.limits.max_total_properties,
        max_enum_values_total=profile.limits.max_enum_values_total,
        max_enum_string_budget_threshold=profile.limits.max_enum_string_budget_threshold,
        max_enum_total_string_length=profile.limits.max_enum_total_string_length,
        max_schema_string_length=32,
    )
    monkeypatch.setitem(
        module._PROFILES,
        target,
        module._ProviderProfile(
            target=profile.target,
            limits=limits,
            require_all_properties=profile.require_all_properties,
        ),
    )
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(_schema(), target=target)
    assert failure.value.code == "schema_size_limit_exceeded"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_large_enums_are_bounded_by_their_string_budget(
    monkeypatch: pytest.MonkeyPatch,
    target: SchemaTarget,
) -> None:
    from penampakan.llms import schema as module

    profile = module._PROFILES[target]
    limits = module._ProviderLimits(
        api_version=profile.limits.api_version,
        max_nesting_depth=profile.limits.max_nesting_depth,
        max_total_properties=profile.limits.max_total_properties,
        max_enum_values_total=1_000,
        max_enum_string_budget_threshold=2,
        max_enum_total_string_length=8,
        max_schema_string_length=profile.limits.max_schema_string_length,
    )
    monkeypatch.setitem(
        module._PROFILES,
        target,
        module._ProviderProfile(
            target=profile.target,
            limits=limits,
            require_all_properties=profile.require_all_properties,
        ),
    )
    values: JsonValue = ["alpha", "bravo", "charlie", "delta"]
    with pytest.raises(ConfigurationError) as failure:
        compile_action_schema(
            _union(_branch("a", value={"enum": values}), _branch("b")), target=target
        )
    assert failure.value.code == "schema_enum_limit_exceeded"


@pytest.mark.parametrize("target", _STRICT_TARGETS)
def test_mixed_scalar_enums_and_annotations_are_preserved(target: SchemaTarget) -> None:
    values: JsonValue = ["a", 1, 1.5, True, None]
    compiled = compile_action_schema(
        _union(
            _branch("a", value={"enum": values, "description": "Mixed scalars."}),
            _branch("b"),
        ),
        target=target,
    )
    branch = next(
        item
        for item in _branches(compiled)
        if cast(dict[str, JsonValue], item["properties"]).get("type") == {"const": "a"}
    )
    value = cast(dict[str, JsonValue], cast(dict[str, JsonValue], branch["properties"])["value"])
    assert value["enum"] == values
    assert value["description"] == "Mixed scalars."


def test_post_validation_reports_unresolved_and_recursive_references() -> None:
    schema: dict[str, JsonValue] = {"$ref": "#/$defs/Absent", "$defs": {}}
    assert validate_action_instance({}, schema) == ("$:$ref",)
    recursive: dict[str, JsonValue] = {
        "$ref": "#/$defs/Node",
        "$defs": {"Node": {"$ref": "#/$defs/Node"}},
    }
    assert validate_action_instance({}, recursive) == ("$:$ref",)


def test_post_validation_supports_type_lists_and_exclusive_bounds() -> None:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": {"type": ["string", "null"]},
            "ratio": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        },
        "required": ["value", "ratio"],
    }
    assert validate_action_instance({"value": None, "ratio": 0.5}, schema) == ()
    assert validate_action_instance({"value": 1, "ratio": 0.5}, schema) == ("$.value:type",)
    assert validate_action_instance({"value": "x", "ratio": 0.0}, schema) == (
        "$.ratio:exclusiveMinimum",
    )
    assert validate_action_instance({"value": "x", "ratio": 1.0}, schema) == (
        "$.ratio:exclusiveMaximum",
    )


def test_prune_optional_nulls_leaves_unmatched_or_scalar_values_untouched() -> None:
    schema = _library_schema()
    # An action whose discriminators match no branch is returned unchanged so the
    # post-validation step, not the pruner, reports the failure.
    unmatched: JsonValue = {"type": "unknown", "answer": None}
    assert prune_optional_nulls(unmatched, schema) == unmatched
    assert prune_optional_nulls("text", schema) == "text"
    array_schema: dict[str, JsonValue] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        },
    }
    pruned = prune_optional_nulls([{"a": "x", "b": None}], array_schema)
    assert pruned == [{"a": "x"}]
