import json
import re
from pathlib import Path

import pytest

DATA = Path(__file__).parents[1] / "tensorfold_web" / "data"
REPO = Path(__file__).parents[2]
UPSTREAM = re.compile(r"^https://github\.com/ashhart/TensorFold/blob/[0-9a-f]{40}/([^#]+)(?:#L(\d+))?$")
BACKEND_NAMES = {"metal": "mlx", "cuda": "cuda", "hip": "hip"}


def load(name):
    return json.loads((DATA / name).read_text())


def kind_of(value):
    """The JSON Schema type names a Python value satisfies."""
    if value is None:
        return {"null"}
    if isinstance(value, bool):
        return {"boolean"}
    if isinstance(value, int):
        return {"integer", "number"}
    if isinstance(value, float):
        return {"number"}
    return {str: {"string"}, list: {"array"}, dict: {"object"}}[type(value)]


def check(schema, value, root, where="$"):
    """Errors of value against the subset of JSON Schema 2020-12 that models.schema.json uses."""
    if "$ref" in schema:
        name = schema["$ref"].removeprefix("#/$defs/")
        errors = check(root["$defs"][name], value, root, where)
        rest = {k: v for k, v in schema.items() if k not in ("$ref", "description")}
        return errors + (check(rest, value, root, where) if rest else [])
    errors = []
    types = schema.get("type")
    if types is not None and not kind_of(value) & ({types} if isinstance(types, str) else set(types)):
        return [f"{where}: {value!r} is not {types}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: {value!r} is not {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {value!r} not in {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{where}: empty")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{where}: {value!r} does not match {schema['pattern']}")
    if kind_of(value) & {"number"} and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{where}: {value} < {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{where}: {value} <= {schema['exclusiveMinimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{where}: fewer than {schema['minItems']} items")
        for i, item in enumerate(value):
            errors += check(schema.get("items", {}), item, root, f"{where}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errors += [f"{where}: missing {key}" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{where}: unknown key {key}" for key in value if key not in props]
        for key, sub in props.items():
            if key in value:
                errors += check(sub, value[key], root, f"{where}.{key}")
    if "anyOf" in schema and all(check(sub, value, root, where) for sub in schema["anyOf"]):
        errors.append(f"{where}: matches none of anyOf")
    for sub in schema.get("allOf", []):
        errors += check(sub, value, root, where)
    if "if" in schema:
        branch = "then" if not check(schema["if"], value, root, where) else "else"
        errors += check(schema.get(branch, {}), value, root, where)
    return errors


def rules(catalog):
    """Rules across records the schema cannot state: ids, references and support."""
    errors = []
    classes = [c["id"] for c in catalog["hardware_classes"]]
    models = {m["id"]: m for m in catalog["models"]}
    for name, ids in (("class", classes), ("model", [m["id"] for m in catalog["models"]]),
                      ("run", [r["id"] for r in catalog["runs"]])):
        errors += [f"duplicate {name} id {i}" for i in set(ids) if ids.count(i) > 1]
    for model in catalog["models"]:
        errors += [f"{model['id']}: unknown class {c}" for s in model["support"] for c in s["hardware"] if c not in classes]
    for run in catalog["runs"]:
        model = models.get(run["model_id"])
        if model is None:
            errors.append(f"{run['id']}: unknown model {run['model_id']}")
            continue
        if not any(s["backend"] == run["backend"] and run["hardware_class"] in s["hardware"] for s in model["support"]):
            errors.append(f"{run['id']}: {model['id']} lists no {run['backend']} support on {run['hardware_class']}")
    return errors


def problems(catalog):
    return check(load("models.schema.json"), catalog, load("models.schema.json")) + rules(catalog)


def test_the_models_catalog_passes_its_schema_and_rules():
    assert problems(load("models.json")) == []


def test_the_schema_check_refuses_each_planted_error():
    good = load("models.json")
    aggregate = next(i for i, r in enumerate(good["runs"]) if r["rate_scope"] == "aggregate")
    single = next(i for i, r in enumerate(good["runs"]) if r["rate_scope"] == "per-stream")
    plants = {
        "a total over the whole request": ("runs", aggregate, "rate_window", "end-to-end"),
        "a total of one stream": ("runs", aggregate, "concurrency", 1),
        "a rate with no scope": ("runs", single, "rate_scope", None),
        "a misspelt key": ("runs", single, "decode_tsp", 1.0),
        "a zero rate": ("runs", single, "decode_tps", 0),
        "an unknown status": ("models", 0, "status", "beta"),
        "an unknown model": ("runs", single, "model_id", "no-such-model"),
        "an unknown class": ("runs", single, "hardware_class", "tpu"),
    }
    for name, (table, index, key, value) in plants.items():
        catalog = json.loads(json.dumps(good))
        target = catalog[table][index]["support"][0] if key == "status" else catalog[table][index]
        target[key] = value
        assert problems(catalog), name


def test_the_schema_check_agrees_with_jsonschema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(load("models.schema.json")).validate(load("models.json"))


def test_every_upstream_receipt_exists_and_carries_its_numbers():
    for run in load("models.json")["runs"]:
        match = UPSTREAM.match(run["receipt_url"])
        if match is None:
            assert run["receipt_url"].startswith("https://github.com/"), run["id"]
            continue
        receipt = REPO / match.group(1)
        assert receipt.is_file(), run["id"]
        text = receipt.read_text()
        for key in ("decode_tps", "prefill_tps"):
            if run[key] is not None:
                assert f"{run[key]:g}" in text, (run["id"], key)


def test_models_agree_with_the_web_catalog():
    entries = {e["id"]: e for e in load("catalog.json")["models"]}
    for model in load("models.json")["models"]:
        entry = entries.get(model["checkpoint"])
        if entry is None:
            continue
        assert entry["family"] == model["family"], model["id"]
        assert {BACKEND_NAMES[s["backend"]] for s in model["support"]} <= set(entry["backends"]), model["id"]
