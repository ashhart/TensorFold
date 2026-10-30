"""What Slate's compile sends to the model (app/slate.py): python3 tests/test_slate_compile.py. No GPU, no network."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import negation as N  # noqa: E402
import server  # noqa: E402  (attaches Slate to the Studio's prompt builders)
import slate  # noqa: E402


def film(**shot) -> dict:
    """One scene with one shot: Leo and Mia on the walkway, as a production run wrote them."""

    one = {"id": "shot-1", "name": "1.4 First Smiles", "intent": "0:12.1-0:16.1 (2 bars, 5 s). The audience sees the first spark.",
           "spec": {"durationSec": 5, "fps": 24, "aspectRatio": "16:9", "size": "MS", "angle": "eye level", "lens": "50mm", "movement": "locked"},
           "prompt": "# Subject\nLeo hands Mia the stack of books and they smile.\n# Composition\nMedium two-shot in profile.\n# Lighting\nLate-morning sun through leaves.",
           "beatSheet": None, "mutedLines": [], "takes": [],
           "tf": {"start": "text", "cast": ["char-leo", "char-mia"], "location": "loc-1", "lines": [], "soundscape": "Leaf rustle and distant birdsong.", "music": "N/A", "cue": "", "seed": 42}}
    one.update(shot)
    return {"id": "p-1", "name": "Hesitation", "defaults": {"durationSec": 8}, "tf": {"res": "480p", "quality": "fh8"},
            "characters": [{"id": "char-leo", "name": "Leo", "age": "21", "gender": "male", "hair": "short sandy-blond hair", "clothing": "a rust-orange work jacket"},
                           {"id": "char-mia", "name": "Mia", "age": "20", "gender": "female", "hair": "long pale-blonde hair", "clothing": "an emerald-green cardigan"}],
            "locations": [{"id": "loc-1", "name": "Campus Walkway", "description": "A paved walkway under old trees"}],
            "scenes": [{"id": "scene-1", "name": "Scene 1", "shots": [one]}]}


def test_a_shots_intent_stays_out_of_the_prompt():
    made = slate.compile_shot(film(), "shot-1")
    assert "0:12.1" not in made["prompt"] and "audience" not in made["prompt"] and "2 bars" not in made["prompt"]
    assert "Leo hands Mia the stack of books" in made["prompt"]  # what happens is the Subject section's
    # with a beat sheet the beats are the action, and the intent is still a note
    beats = slate.compile_shot(film(beatSheet=[{"from": 0, "to": 2, "text": "Leo holds out the books."}, {"from": 2, "to": 5, "text": "Mia takes them and smiles."}]), "shot-1")
    assert "First, leo holds out the books. Finally, mia takes them and smiles." in beats["prompt"] and "audience" not in beats["prompt"]
    # an empty shot still compiles, and says so
    empty = slate.compile_shot(film(prompt=""), "shot-1")
    assert "audience" not in empty["prompt"] and any("no prompt yet" in w for w in empty["warnings"])


def test_a_compiled_prompt_forbids_nothing_by_name():
    """Slate's own words say what is there, so the negation check lists nothing in a prompt from a clean shot."""

    had, N.OWN = N.OWN, ()
    try:
        made = slate.compile_shot(film(), "shot-1")
        prompt = made["prompt"]
        assert "Static shot on a locked, still camera." in prompt and "does not move" not in prompt
        assert "Mouths stay closed." in prompt and "Only ambient sound is heard." in prompt
        assert "Nobody speaks" not in prompt and "No voices" not in prompt
        assert N.find(prompt.replace("non_diegetic_music: N/A", "")) == [] and N.check(prompt)["negations"] == []
        assert not any("no negative prompt" in w for w in made["warnings"])
        for move in slate.MOVES.values():
            assert N.find(move) == [], move
        # a shot with nobody cast and no person in its words: no mouth is named into it
        empty = film(prompt="# Subject\nWaves roll onto an empty beach at dusk.", tf={"start": "text", "cast": [], "location": "", "lines": [], "soundscape": "", "music": "N/A", "seed": 1})
        assert "Mouths" not in slate.compile_shot(empty, "shot-1")["prompt"]
        # the director's own negation is still said, never rewritten
        told = slate.compile_shot(film(prompt="# Subject\nLeo walks. No cars anywhere."), "shot-1")
        assert "No cars anywhere." in told["prompt"] and any("No cars" in w for w in told["warnings"])
    finally:
        N.OWN = had


def music_video() -> dict:
    """"Hesitation": 117.45 BPM, the first beat at 0.07 s, a bar of 2.043 s; three shots of 2, 4 and 2 bars."""

    made = film()
    made["tf"]["song"] = {"name": "Hesitation, kAi", "bpm": 117.45, "beatsPerBar": 4, "firstBeat": 0.07}
    first = made["scenes"][0]["shots"][0]
    first["tf"]["bars"] = 2
    for index, bars in ((2, 4), (3, 2)):
        more = {**first, "id": f"shot-{index}", "name": f"1.{index}", "tf": {**first["tf"], "bars": bars}}
        made["scenes"][0]["shots"].append(more)
    return made


def test_a_slot_in_bars_becomes_the_clip_length_that_covers_it():
    made = music_video()
    song = slate.song_of(made)
    assert round(song["bar"], 3) == 2.043 and song["beats"] == 4
    one = slate.compile_shot(made, "shot-1")["plan"]
    # 2 bars are 4.087 s, 98 frames: the shortest count that covers them is 107 (the run rendered 124 and trimmed)
    assert one["slot"] == {"bars": 2.0, "seconds": 4.087, "frames": 98, "start": 0.07, "bar": 1.0, "spare": 9} and one["frames"] == 107
    assert one["seconds"] == round(107 / 24, 3)
    two = slate.compile_shot(made, "shot-2")["plan"]
    # 4 bars are 8.17 s, 196 frames: 8 seconds (192 frames) is 4 short, the run took a 10 s clip; 209 frames cover it
    assert two["slot"]["frames"] == 196 and two["frames"] == 209 and two["slot"]["spare"] == 13
    assert two["slot"]["bar"] == 3.0 and two["slot"]["start"] == round(0.07 + 2 * song["bar"], 3)
    three = slate.compile_shot(made, "shot-3")["plan"]
    assert three["slot"]["bar"] == 7.0 and three["slot"]["start"] == round(0.07 + 6 * song["bar"], 3)
    # every length a slot gets is one the model takes, and the job is built from it
    for plan in (one, two, three):
        assert (plan["frames"] - 5) % 17 == 0 and server.clip_frames(plan) == plan["frames"]
    # a shot without a slot keeps its seconds, and the shots after it are placed after its clip
    made["scenes"][0]["shots"][0]["tf"]["bars"] = None
    assert "slot" not in slate.compile_shot(made, "shot-1")["plan"] and slate.compile_shot(made, "shot-1")["plan"]["frames"] == 124
    assert slate.compile_shot(made, "shot-2")["plan"]["slot"]["start"] == round(0.07 + 124 / 24, 3)
    # bars without a song, and a slot longer than a clip, are said
    alone = film()
    alone["scenes"][0]["shots"][0]["tf"]["bars"] = 2
    told = slate.compile_shot(alone, "shot-1")
    assert "slot" not in told["plan"] and any("no song" in w for w in told["warnings"])
    made["scenes"][0]["shots"][1]["tf"]["bars"] = 8
    long = slate.compile_shot(made, "shot-2")
    assert long["plan"]["frames"] == 362 and any("longer than a clip" in w for w in long["warnings"])
    # a tempo that is none is no song
    for bad in ({}, {"bpm": 0}, {"bpm": "fast"}, {"bpm": 5000}, {"bpm": 120, "beatsPerBar": 0}):
        assert slate.song_of({"tf": {"song": bad}}) is None or bad.get("beatsPerBar") == 0 and slate.song_of({"tf": {"song": bad}})["beats"] == 4


def test_a_shots_own_lora_goes_into_its_plan_and_through_the_memory_guard():
    made = music_video()
    made["scenes"][0]["shots"][0]["tf"]["loras"] = [{"id": "camera-motion", "strength": 0.9, "style": "Orbit"}, {"id": ""}, "junk"]
    plan = slate.compile_shot(made, "shot-1")["plan"]
    assert plan["loras"] == [{"id": "camera-motion", "strength": 0.9, "style": "Orbit"}] and "loras" not in slate.compile_shot(made, "shot-2")["plan"]
    # the guard's verdict is the queue's own: asked with the plan, and with whether a LoRA is on the shot
    asked, had = [], slate.room
    try:
        slate.room = lambda plan, lora: asked.append((plan["frames"], plan.get("preset"), lora)) or {"fits": not lora, "reason": "a clip with a LoRA ... is refused. What fits: 1280x736 at this length.", "need_mb": 76843, "nearest": {"size": [1280, 736]}}
        made["tf"]["res"] = "sharp"
        told = slate.compile_shot(made, "shot-1")
        assert asked[-1] == (107, "sharp", True) and told["memory"]["fits"] is False and told["memory"]["nearest"] == {"size": [1280, 736]}
        assert any(w.startswith("This shot will be refused when queued:") and "What fits" in w for w in told["warnings"])
        fine = slate.compile_shot(made, "shot-2")
        assert asked[-1] == (209, "sharp", False) and fine["memory"]["fits"] is True and not any("refused" in w for w in fine["warnings"])
        # the film's LoRAs count on a shot that has none of its own
        made["tf"]["loras"] = [{"id": "camera-motion"}]
        slate.compile_shot(made, "shot-2")
        assert asked[-1][2] is True
    finally:
        slate.room = had
    # on a DGX Spark a Sharp shot with a LoRA at 124 frames fits where the engine merges the LoRA, else is refused
    if server.SPARK:
        import memory_guard

        was = memory_guard.room_mb
        memory_guard.room_mb = lambda: 118700
        try:
            made["tf"]["loras"] = []
            made["scenes"][0]["shots"][0]["tf"]["bars"] = None
            made["scenes"][0]["shots"][0]["spec"]["durationSec"] = 5
            real = slate.compile_shot(made, "shot-1")
            merges = server.SPARK_ENGINE_LIB.exists() and memory_guard.engine_takes_lora(server.SPARK_ENGINE_LIB)
            assert real["plan"]["frames"] == 124
            assert real["memory"]["fits"] is True if merges else (real["memory"]["fits"] is False and "1280x736 at this length" in real["memory"]["reason"])
            made["tf"]["res"] = "720p"
            assert slate.compile_shot(made, "shot-1")["memory"]["fits"] is True
        finally:
            memory_guard.room_mb = was


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
