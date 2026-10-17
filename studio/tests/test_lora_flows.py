"""app/lora_flows.py: what a flow LoRA needs of a clip and the prompt it is given."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import lora_flows as F  # noqa: E402


def presets(rows):
    path = Path(tempfile.mkdtemp()) / "lora-presets.json"
    path.write_text(json.dumps(rows))
    F.PRESETS = path


ROWS = [{"id": "character-swap", "flow": "character-swap"}, {"id": "lineart-anime", "flow": "lineart"},
        {"id": "keyframe-animation", "flow": "keyframe"}, {"id": "anime-slider", "flow": "slider"},
        {"id": "camera-motion"}, {"id": "odd", "flow": "no-such-flow"}]


def test_a_preset_names_its_flow_and_the_rest_have_none():
    presets(ROWS)
    assert F.flow_of("character-swap") == "character-swap" and F.flow_of("camera-motion") == "" and F.flow_of("odd") == "" and F.flow_of("nobody") == ""
    assert F.picked([{"id": "camera-motion"}, {"id": "lineart-anime", "strength": 1}]) == [("lineart-anime", "lineart")]
    assert F.public("camera-motion") is None and F.public("lineart-anime")["lineart"] is True


def test_what_each_flow_needs_of_the_clip():
    presets(ROWS)
    swap, line, key = [{"id": "character-swap"}], [{"id": "lineart-anime"}], [{"id": "keyframe-animation"}]
    assert F.fits(swap, 1, 1, True) is None and F.fits(swap, 1, 1, False) is None
    assert "source clip" in F.fits(swap, 1, 0, True) and "picture" in F.fits(swap, 0, 1, True)
    assert F.fits(line, 0, 1, False) is None and "line-art video" in F.fits(line, 0, 0, False)
    assert "no reference picture" in F.fits(line, 1, 1, False)
    assert F.fits(key, 2, 0, False) is None and "20 passes" in F.fits(key, 1, 0, True) and "at most 2" in F.fits(key, 3, 0, False)
    # a LoRA with no flow, and the slider, ride on any clip
    assert F.fits([{"id": "camera-motion"}, {"id": "anime-slider"}], 0, 0, False) is None
    assert F.fits([{"id": "anime-slider"}], 3, 1, True) is None


def test_plain_prompts_and_the_authors_caption_styles():
    presets(ROWS)
    assert F.plain([{"id": "lineart-anime"}]) and not F.plain([{"id": "character-swap"}]) and not F.plain([])
    swap = F.compose([{"id": "character-swap"}], {"video_subjects": ["the man in the purple shirt."], "subjects": ["a knight in silver armour"]})
    assert swap.startswith("Replace only the man in the purple shirt in <Video 1> with the character in <Picture 1>, a knight in silver armour.")
    assert "Do not show the reference sheet or its background." in swap
    assert F.compose([{"id": "character-swap"}], {}).startswith("Replace only the person in <Video 1>")
    assert F.compose([{"id": "lineart-anime"}], {"scene": "A girl with long blue hair by an ice wall", "action": "she looks up"}) == "A girl with long blue hair by an ice wall. she looks up."
    assert F.compose([{"id": "lineart-anime"}], {}) == "Generate anime videos from line art outlines."
    key = F.compose([{"id": "keyframe-animation"}], {"scene": "a fox, clean line on white", "action": "it leaps from a crouch to full stretch"})
    assert key.startswith("Create the next keyframe in the sequence. SCENE: a fox, clean line on white. MOVEMENT: it leaps") and "STATIC:" in key
    assert F.compose([{"id": "camera-motion"}, {"id": "anime-slider"}], {"scene": "x"}) is None


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
