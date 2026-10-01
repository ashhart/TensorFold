"""Necessary helper boundary checks: no service/model operations."""
from unittest.mock import patch
import pytest
from tensorfold.families.deepseek_v4.cuda import build_runtime as b


def args(tmp_path, *extra):
    return b.parser().parse_args(['--gguf', str(tmp_path / 'target.gguf'),
        '--model-dir', str(tmp_path / 'model'), '--companion-reserve-gib', '0', *extra])


def test_preflight_read_only_and_invalid_budget_before_inspection(tmp_path):
    with patch.object(b, 'inspect_inputs', return_value={'artifact_valid': True}), \
         patch.object(b, 'install_candidate', side_effect=AssertionError('installation reached')):
        assert b.execute(args(tmp_path, '--preflight-only'))['artifact_valid']
        assert not (tmp_path / 'model').exists()
    for option in ('nan', 'inf', '-1'):
        with patch.object(b, 'inspect_inputs', side_effect=AssertionError('inspection reached')):
            with pytest.raises(ValueError):
                b.execute(args(tmp_path, '--companion-reserve-gib', option))


def test_install_failure_leaves_no_prepared_model(tmp_path):
    report = {'artifact_valid': True}
    with patch.object(b, 'inspect_inputs', return_value=report), \
         patch.object(b, 'install_candidate', side_effect=RuntimeError('wheel failed')):
        with pytest.raises(RuntimeError, match='wheel failed'):
            b.execute(args(tmp_path))
    assert not (tmp_path / 'model/config.json').exists()


def test_success_and_preparation_publish_installed_library(tmp_path):
    library = tmp_path / 'installed.so'
    source = tmp_path / 'target.gguf'
    source.write_bytes(bytes(100))
    report = {'artifact_valid': True, 'source': str(tmp_path / 'target.gguf'),
              'arch': {}, 'source_size': 100, 'header_sha256': 'header',
              'source_identity': {**b.stamp(source), 'sha256_scope': 'gguf-header'},
              'tokenizer': {}, 'external_tokenizer': {}}
    with patch.object(b, 'inspect_inputs', return_value=report), \
         patch.object(b, 'install_candidate', return_value=(library, {'sha256': 'wheel'})), \
         patch.object(b, 'prepare_candidate', side_effect=lambda model, **_: model.mkdir()) as prepare:
        result = b.execute(args(tmp_path))
    assert prepare.call_args.kwargs['native_library'] == library
    assert prepare.call_args.kwargs['source_identity']['sha256_scope'] == 'gguf-header'
    assert result['wheel']['sha256'] == 'wheel'
