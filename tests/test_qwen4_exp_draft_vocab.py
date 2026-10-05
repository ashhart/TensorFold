"""The shipped Qwen3.8 Flash Next draft vocabularies: what the names resolve to and how they nest."""

import pytest
@pytest.mark.torch
def test_the_shipped_cjk_draft_list_extends_the_default_one():
    pytest.importorskip("numpy")

    from tensorfold.families.qwen4_exp.cuda.weight_types import DRAFT_VOCABS, draft_token_ids

    default = draft_token_ids("default")
    cjk = draft_token_ids("cjk")
    assert set(DRAFT_VOCABS) == {"default", "cjk"}
    assert len(default) == 79_591 and len(cjk) == 135_040
    assert list(cjk) == sorted(set(cjk))                    # sorted, no duplicates
    assert set(default) <= set(cjk)                          # the extension keeps every shipped id
    assert draft_token_ids(None) is None and len(draft_token_ids(1024)) == 1024
