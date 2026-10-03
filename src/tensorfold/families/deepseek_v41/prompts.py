"""Chat prompts: V4.1 keeps V4's message format for text (DSML tool tags change; see the recipe's notes)."""

from __future__ import annotations

from tensorfold.families.deepseek_v4.prompts import DeepSeekTokenizer


class DeepSeekV41Tokenizer(DeepSeekTokenizer):
    """The checkpoint's tokenizer with V4's encoder until V4.1's (DSML tags, numeric effort) is vendored."""
