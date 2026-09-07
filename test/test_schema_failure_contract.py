import pytest

from v_core.learning.schema import SchemaError, validate_instance, validate_schema


@pytest.mark.parametrize("schema", [
    {"type": ["string", "null"]},
    {"type": {"unexpected": "object"}},
    {"type": "object", 1: True, "unknown": True},
    {"type": "object", "properties": {"nested": {"type": []}}},
])
def test_malformed_schema_has_controlled_error(schema):
    with pytest.raises(SchemaError):
        validate_schema(schema)


@pytest.mark.parametrize("allow_extra", [False, True])
def test_non_json_object_keys_rejected_as_contract_error(allow_extra):
    with pytest.raises(SchemaError, match="object keys must be strings"):
        validate_instance(
            {1: "value", "other": "value"},
            {"type": "object", "additionalProperties": allow_extra},
        )
