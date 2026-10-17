"""app/negation.py: what a video prompt forbids by name is listed for the user; the prompt itself is never changed."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import negation as N  # noqa: E402
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
phrases = lambda text: [one["phrase"] for one in N.find(text)]


def test_the_three_ways_of_forbidding_a_thing():
    assert phrases("A bedroom with no photographs.") == ["no photographs"]
    assert phrases("One take without any cut.") == ["without any cut"]
    assert phrases("She does not smile.") == ["not smile"]
    assert phrases("They NEVER look at the camera, and don't turn round.") == ["NEVER look at the camera", "don't turn round"]
    assert N.find("No close-up of her face.")[0]["thing"] == "close-up face"
    # the production run's own line, which hung the wall with photographs
    assert "No photographs or portraits" in phrases("She turns from the window. No photographs or portraits appear anywhere.")


def test_what_is_not_a_negation_is_left_alone():
    assert phrases("A notable nobleman ties a knot north of Nottingham.") == []
    assert phrases("The fox is not only quick but quiet; no doubt it is night.") == []
    assert phrases("A man walks his dog on the beach at noon.") == []
    # each phrase once
    assert phrases("no text. A sign. No text.") == ["no text"]


def test_the_studios_own_sentences_forbid_nothing_by_name():
    """The app's own words for a quiet clip and a locked camera say what is there: the check finds nothing in them."""

    had, N.OWN = N.OWN, ()
    try:
        prompt = server.compose({"scene": "A quiet harbour at dawn.", "action": "A boat leaves."}, False)
        assert "Nobody speaks" not in prompt and "No voices" not in prompt and "Only ambient sound is heard." in prompt
        assert "Mouths" not in prompt  # nobody is in the shot: a mouth is not named into it
        assert N.find(prompt.replace("non_diegetic_music: N/A", "")) == []
        people = server.compose({"scene": "A woman waits on the quay.", "action": "She watches a boat leave."}, False)
        assert "Mouths stay closed." in people and N.find(people.replace("non_diegetic_music: N/A", "")) == []
        spoken = server.compose({"scene": "A kitchen.", "dialogue": [{"text": "Good morning."}]}, False)
        assert "stays silent" in spoken and "After the spoken lines only ambient sound is heard." in spoken
        assert N.find(spoken.replace("non_diegetic_music: N/A", "").replace("Good morning.", "")) == []
        # a soundscape that already says it, in the old words or the new, is not told twice
        assert server.compose({"scene": "A hall.", "soundscape": "Rain. No speech."}, False).count("ambient sound") == 0
        assert server.compose({"scene": "A hall.", "soundscape": "Rain. Only ambient sound is heard."}, False).count("Only ambient sound") == 1
        # a clip from a reference picture: its subject is a person unless the fields say otherwise
        ref = server.compose_references({"scene": "A cafe.", "subjects": ["a woman in a red coat"]}, 1)
        assert "Mouths stay closed." in ref and "Only ambient sound is heard." in ref and N.find(ref) == []
        assert "Mouths" not in server.compose_references({"scene": "A cafe.", "subjects": ["a red teapot"], "people": False}, 1)
    finally:
        N.OWN = had
    # prompts written before this change still pass: the old sentences are passed over
    assert N.check("integrated_multimodal_description: [Shot 1] A hall. Nobody speaks.\n\noverall_soundscape: Rain. No voices, no speech.")["negations"] == []
    # the soundscape and the music are the sound track's own fields
    assert N.check("integrated_multimodal_description: [Shot 1] A hall.\n\noverall_soundscape: Rain, no thunder.\n\nnon_diegetic_music: N/A")["negations"] == []


def test_compose_lists_them_by_box_and_sends_the_prompt_as_written():
    fields = {"scene": "A dark bedroom with no photographs on the walls.", "camera": "A steady shot.", "action": "She never looks at the camera."}
    reply = client.post("/api/compose", json={"fields": fields, "first_frame": False}).json()
    assert [(one["phrase"], one["field"]) for one in reply["negations"]] == [("no photographs on the walls", "Scene"), ("never looks at the camera", "Action")]
    assert "no negative prompt" in reply["reason"]
    assert "no photographs on the walls" in reply["prompt"] and "never looks at the camera" in reply["prompt"]  # nothing rewritten
    clean = client.post("/api/compose", json={"fields": {"scene": "A dark bedroom with plain walls."}, "first_frame": False}).json()
    assert clean["negations"] == [] and clean["reason"] == ""
    # the reference form goes through the same check
    reply = client.post("/api/compose", json={"fields": {"scene": "A street without cars."}, "references": 1}).json()
    assert [one["phrase"] for one in reply["negations"]] == ["without cars"]


def test_a_prompt_written_by_hand_is_checked_before_it_is_queued():
    reply = client.post("/api/prompt/check", json={"prompt": "[Shot 1] A hall. No cut, no close-up. Nobody speaks.\n\noverall_soundscape: No voices, no speech."}).json()
    assert [one["phrase"] for one in reply["negations"]] == ["No cut", "no close-up"] and reply["reason"]
    assert client.post("/api/prompt/check", json={"prompt": "A hall at dusk."}).json() == {"negations": [], "reason": ""}


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
