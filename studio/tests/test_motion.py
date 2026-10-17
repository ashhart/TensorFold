"""Motion Previs as a reference: the hand-off wording, the pace rules and the tab's API; no GPU, models or network."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(tempfile.mkdtemp()) / "TensorFoldStudio"
os.environ["STUDIO_HOME"] = str(HOME)
sys.path.insert(0, str(ROOT / "app"))
import motion_prompt  # noqa: E402
import pace  # noqa: E402
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
SLOW = {"motion": "slow", "sound": "impacts"}


def test_the_source_clip_with_a_new_person_is_declared_as_footage_with_its_person_replaced():
    made = motion_prompt.motion_template({"layer": "reference", "mode": "full-scene"}, 1)
    assert made["lead"].startswith("The target video is <Video 1> with its person replaced by <Subject 1>")
    assert "first-frame anchor" not in " ".join(made["video_roles"] + made.get("definitions", []))
    two = motion_prompt.motion_template({"layer": "reference", "mode": "full-scene", "people": 2}, 1)
    assert "its people are replaced by" in two["video_roles"][0] and "move as those people do" in two["video_follows"][0]


def test_a_control_layer_gets_a_legend_and_is_never_the_look_of_the_clip():
    made = motion_prompt.motion_template({"layer": "depth", "mode": "camera-only"}, 1)
    assert any("Legend of <Video 1>: a depth map" in line for line in made["definitions"])
    assert any("labels only" in line for line in made["definitions"])
    assert "no skeleton lines" in made["avoid"] and made["video_follows"] == ["The camera follows the path of <Video 1> frame by frame."]


def test_the_reference_prompt_carries_the_hand_off():
    fields = {"scene": "A dojo at dusk.", "subjects": ["a woman in a grey gi"], "video_subjects": ["the drill"],
              "motion": {"layer": "depth", "mode": "actor-motion", "camera": "The camera pans left slowly."}}
    prompt = server.compose_references(fields, 1)
    assert "<Video 1> is the structural template of the target video" in prompt
    assert "The camera pans left slowly." in prompt and "labels only" in prompt
    # without a hand-off the same reference is an ordinary motion video
    plain = server.compose_references({**fields, "motion": None}, 1)
    assert "structural template" not in plain and "<Video 1>" in plain


def test_impacts_over_a_slow_reference_leave_the_video_out():
    motion = {"layer": "reference", "mode": "full-scene", "free": True, "measured": SLOW}
    assert motion_prompt.freed("impacts", motion, 1)
    assert not motion_prompt.freed("impacts", motion, 0)  # without a picture nothing carries who is in it
    assert not motion_prompt.freed("impacts", {**motion, "measured": {"motion": "fast"}}, 1)
    prompt = server.compose_references({"scene": "A yard.", "subjects": ["a man"], "video_subjects": [""], "pace": "impacts",
                                        "motion": motion, "action": "He strikes the dummy."}, 1)
    assert "<Video 1>" not in prompt and "full speed" in prompt


def test_flow_takes_the_hits_out_and_automatic_steps_in_only_over_a_slow_reference():
    fields = {"action": "They spar.", "soundscape": "Sharp slaps of hands, a fan humming.", "music": "N/A"}
    flow = motion_prompt.paced_fields(fields, {"measured": SLOW}, True, 1)
    assert flow["pace"] == "flow" and not pace.has_impacts(flow["fields"]["soundscape"]) and pace.SMOOTH in flow["fields"]["action"]
    kept = motion_prompt.paced_fields(fields, {"measured": {"motion": "fast", "sound": "impacts"}}, True, 1)
    assert kept == {"fields": fields, "pace": ""}
    assert pace.choose("auto", {"motion": "fast"}) == "impacts" and pace.choose("flow", {"motion": "fast"}) == "flow"
    assert pace.choose("auto", None) == "" and pace.calm("Slaps, a fan humming.") == "A fan humming."


def test_the_tab_answers_without_its_environment():
    state = client.get("/api/motion/state").json()
    assert set(state["modes"]) == {"camera-only", "actor-motion", "object-motion", "full-scene"}
    assert set(state["available"]) == {"ffmpeg", "pose", "camera", "ai_depth"}
    assert client.get("/api/motion/handoff").json() == {"reference": None}
    assert client.post("/api/motion/reference", json={}).status_code == 400


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
