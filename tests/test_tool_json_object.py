import json
import math

import pytest

from tensorfold.server.tools import _tool_json_object


@pytest.mark.parametrize("value", [None, "", " \t\r\n", "\u00a0", "\u2003\u00a0"])
def test_absent_or_unicode_blank_arguments_make_a_new_empty_object(value):
    first = _tool_json_object(value)
    assert first == {}
    assert first is not _tool_json_object(value)


@pytest.mark.parametrize("text", ['{"limit": 1}', '\u00a0\u2003{"limit": 1}\u2003\u00a0'])
def test_json_arguments_strip_unicode_whitespace_before_parsing(text):
    assert _tool_json_object(text) == {"limit": 1}


def test_dictionary_arguments_keep_their_identity_and_values():
    arguments = {"second": [], "first": float("nan")}
    assert _tool_json_object(arguments) is arguments
    assert list(arguments) == ["second", "first"]


@pytest.mark.parametrize("value", [[], (), 1, 1.5, True, b"{}", object(),
                                 "[]", "null", "true", "1", "1.5", '"text"', "\u00a0[]\u00a0"])
def test_non_object_arguments_keep_the_same_error(value):
    with pytest.raises(ValueError) as error:
        _tool_json_object(value)
    assert type(error.value) is ValueError
    assert str(error.value) == "tool_call arguments must be a JSON object"


@pytest.mark.parametrize("text", ['{"limit":', "not JSON", "{\u00a0}"])
def test_malformed_arguments_keep_the_json_decoder_error(text):
    with pytest.raises(json.JSONDecodeError) as reference:
        json.loads(text.strip())
    with pytest.raises(json.JSONDecodeError) as error:
        _tool_json_object(text)
    assert (error.value.msg, error.value.doc, error.value.pos) == (
        reference.value.msg, reference.value.doc, reference.value.pos)


def test_json_objects_keep_key_order_and_existing_nonfinite_values():
    arguments = _tool_json_object('{"second": NaN, "first": Infinity, "second": -Infinity}')
    assert list(arguments) == ["second", "first"]
    assert arguments["second"] == -math.inf
    assert arguments["first"] == math.inf
