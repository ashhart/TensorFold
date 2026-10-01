"""TENSORFOLD_SEED_SALT moves every prompt-derived seed; unset or 0 keeps today's seeds."""

import pytest

from tensorfold.engine import exact_sampling


def test_the_default_salt_keeps_the_prompt_seed(monkeypatch):
    monkeypatch.setattr(exact_sampling, "SEED_SALT", 0)
    assert exact_sampling.seed_for([1, 2, 3]) == exact_sampling.seed_for([1, 2, 3], 0)


def test_a_salt_moves_the_prompt_seed_and_stays_reproducible(monkeypatch):
    base = exact_sampling.seed_for([1, 2, 3], 0)
    monkeypatch.setattr(exact_sampling, "SEED_SALT", 2)
    assert exact_sampling.seed_for([1, 2, 3]) != base
    assert exact_sampling.seed_for([1, 2, 3]) == exact_sampling.seed_for([1, 2, 3], 2)


@pytest.mark.parametrize("value,salt", [("", 0), ("0", 0), ("7", 7), (" -3 ", -3)])
def test_the_salt_reads_the_environment(monkeypatch, value, salt):
    monkeypatch.setenv("TENSORFOLD_SEED_SALT", value)
    assert exact_sampling._salt_from_env() == salt


def test_a_malformed_salt_is_refused(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SEED_SALT", "two")
    with pytest.raises(ValueError, match="TENSORFOLD_SEED_SALT"):
        exact_sampling._salt_from_env()
