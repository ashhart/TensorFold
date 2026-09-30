"""Schemas without a usable ``type`` stay untyped: decode_parameter keeps the text and nothing raises."""

import pytest

from tensorfold.tool_parameters import decode_parameter, typed_parameter


@pytest.mark.parametrize("schema,value", [
    ({}, "None"),                          # no type at all
    ({"type": ["integer", "null"]}, "5"),   # a list where a type name belongs
    ({"type": "weird"}, "None"),            # a name the type dict does not name
    ({"type": "string"}, "7.5"),            # a scalar type that takes no decode path
])
def test_untyped_schemas_keep_the_text(schema, value):
    assert not typed_parameter(schema)
    for python in (True, False):
        assert decode_parameter(value, schema, python=python) == value


def test_a_typed_schema_still_decodes():
    assert typed_parameter({"type": "integer"})
    assert decode_parameter("True", {"type": "boolean"}) is True
    assert decode_parameter("('a',)", {"type": "integer"}) == "('a',)"
