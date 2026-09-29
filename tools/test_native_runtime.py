import pytest

from native_runtime import check_requirements, resolved_dependencies


def project(*requirements):
    return {"project": {"dependencies": list(requirements)}}


def test_upstream_mlx_upper_bound():
    upstream = project("mlx>=0.32.2,<0.32.3")
    check_requirements(upstream, {"mlx": "0.32.2"}, {"mlx": "0.32.2"})
    with pytest.raises(RuntimeError, match="installed 0.32.3"):
        check_requirements(upstream, {"mlx": "0.32.3"}, {"mlx": "0.32.2"})


def test_upstream_bump_rejects_old_native_pin_even_with_new_python():
    with pytest.raises(RuntimeError, match="native pin 0.32.2 is incompatible"):
        check_requirements(project("mlx>=0.33"), {"mlx": "0.33.0"}, {"mlx": "0.32.2"})


def test_model_library_pin_is_checked_and_names_are_normalized():
    check_requirements(project("MLX_LM>=0.31.3,<0.32"), {"mlx-lm": "0.31.3"}, {"mlx-lm": "0.31.3"})
    with pytest.raises(RuntimeError, match="native pin 0.31.3 is incompatible"):
        check_requirements(project("mlx-lm>=0.32"), {"mlx-lm": "0.32"}, {"mlx-lm": "0.31.3"})


def test_new_dependency_and_inactive_marker():
    check_requirements(project('missing-package>=1; python_version < "2"'), {}, {})
    with pytest.raises(RuntimeError, match="installed missing"):
        check_requirements(project("new-dependency>=1"), {}, {})


def test_direct_source_requirement_needs_review():
    with pytest.raises(RuntimeError, match="direct-source dependencies"):
        check_requirements(project("mlx @ https://example.invalid/mlx.whl"), {"mlx": "0.32.2"}, {"mlx": "0.32.2"})


def test_upstream_resolution_updates_native_pairing_instead_of_freezing_it():
    previous = {"python": {"mlx": "0.32.2", "mlx-metal": "0.32.2", "mlx-lm": "0.31.3"},
                "mlx_revision": "old-commit", "mlx_c_revision": "bridge-commit"}
    versions = {"mlx": "0.33.1", "mlx-metal": "0.33.1", "mlx-lm": "0.32.2"}
    result = resolved_dependencies(previous, versions, "0.32.2")
    assert result["python"] == versions
    assert result["mlx_revision"] == "v0.33.1"
    assert result["rebuild_mlx"]
    # Retry a partial install: MLX may already be new while its C bridge failed.
    assert resolved_dependencies(previous, versions, "0.33.1")["rebuild_mlx"]
    assert previous["mlx_revision"] == "old-commit"
    assert not resolved_dependencies(previous, previous["python"], "0.32.2")["rebuild_mlx"]
    assert resolved_dependencies(previous, previous["python"], None)["rebuild_mlx"]


def test_resolver_rejects_mismatched_mlx_metal_wheels():
    with pytest.raises(RuntimeError, match="versions disagree"):
        resolved_dependencies({}, {"mlx": "0.33.1", "mlx-metal": "0.32.2"}, None)
