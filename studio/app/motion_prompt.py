"""The Motion Previs hand-off in a reference prompt: a layer of a Motion Previs pack as ``<Video 1>``, and its pace."""

from __future__ import annotations

import pace
from settings import REFERENCE_LIMIT

# What a Motion Previs layer shows, for the model: a control layer is a diagram of the shot, not footage of it.
MOTION_LEGENDS = {
    "openpose": "an OpenPose skeleton on a black background, where the coloured lines are the joints and limbs of the subject's body and the black area is everything else",
    "pose": "a pose skeleton on a black background, where the coloured lines are the joints and limbs of the subject's body and the black area is everything else",
    "ai-depth": "a depth map in shades of grey, where brighter is nearer to the camera and darker is farther away",
    "depth": "a depth map in shades of grey, where brighter is nearer to the camera and darker is farther away",
    "edges": "an edge map, where the bright lines on black are the outlines of the people and objects",
    "lineart": "a line drawing, where the lines are the outlines of the people and objects",
    "masks": "a motion mask, where the bright areas are what moves and the black area is what stays still",
    "normals": "a surface-direction map, where each colour stands for the way a surface faces",
}
MOTION_MODES = ("camera-only", "actor-motion", "object-motion", "full-scene")
# what each mode keeps of a control layer, and of the source clip itself
MOTION_KEEPS = {"camera-only": "the camera path and the scale and layout it gives the scene",
                "actor-motion": "the positions, silhouettes and body movement of its figure and the camera path",
                "object-motion": "the position and motion path of its main object and the camera path",
                "full-scene": "the layout, silhouettes, positions, movement and camera path"}
MOTION_SOURCE = {
    "camera-only": ("the camera reference for the target video: only its camera movement and timing are kept",
                    "the camera movement and timing of <Video 1> are followed; its people, objects and setting are not rendered.",
                    "The camera moves as the camera of <Video 1> does."),
    "actor-motion": ("the motion reference for the target video: the body motion of its person and its camera movement are kept, and who it is and where it is are replaced",
                     "the position, stance, body motion and timing of its person and the camera movement of <Video 1> are followed frame by frame; its person and its setting are not rendered.",
                     "{who} moves as the person in <Video 1> does, move for move, and the camera follows the camera of <Video 1>."),
    "object-motion": ("the motion reference for the target video: the path of its main object and its camera movement are kept, and the object's design and the setting are replaced",
                      "the motion path and timing of the main object and the camera movement of <Video 1> are followed frame by frame; its design, surface and setting are not rendered.",
                      "The main object moves along the path of the object in <Video 1>, and the camera follows the camera of <Video 1>."),
    "full-scene": ("the source video: its setting, its camera movement and the motion in it are kept",
                   "the setting, light, camera path and the position, motion and timing of what moves in <Video 1> are followed frame by frame.",
                   "The scene, the motion and the camera follow <Video 1>."),
}


# the same sentences for a clip with more than one person in it: longest phrases first
MOTION_PLURAL = (
    ("the body motion of its person", "the body motion of its people"), ("its person is replaced by", "its people are replaced by"),
    ("with its person replaced by", "with its people replaced by"), ("the person shown in", "the people shown in"),
    ("the person of <Video 1> is not rendered", "the people of <Video 1> are not rendered"),
    ("its person and its setting are not rendered", "its people and its setting are not rendered"),
    ("<Subject 1> takes the place of the person in <Video 1> and moves as that person does",
     "<Subject 1> take the places of the people in <Video 1>, one for one, and move as those people do"),
    ("moves as the person in <Video 1> does", "move as the people in <Video 1> do"),
    ("timing of its person", "timing of its people"), ("of its figure", "of its figures"),
    ("are those of the figure in <Video 1>", "are those of the figures in <Video 1>, one for one"),
)


def motion_template(motion: dict, pictures: int) -> dict:
    """``motion_template_one``, in the plural when the user says ``<Video 1>`` holds two or more people."""

    made = motion_template_one(motion, pictures)
    try:
        people = int(motion.get("people") or 1)
    except (TypeError, ValueError):
        people = 1
    if people < 2:
        return made

    def plural(text: str) -> str:
        for one, many in MOTION_PLURAL:
            text = text.replace(one, many)
        return text

    return {key: [plural(line) for line in value] if isinstance(value, list) else plural(value) for key, value in made.items()}


def motion_template_one(motion: dict, pictures: int) -> dict:
    """What a Motion Previs layer adds to the Ref2VA prompt: how ``<Video 1>`` is declared, kept and avoided."""

    layer = motion.get("layer") or "reference"
    mode = motion.get("mode") if motion.get("mode") in MOTION_MODES else "full-scene"
    who = "<Subject 1>" if pictures else "The subject"
    if layer not in MOTION_LEGENDS:  # the source clip
        role, keep, follow = MOTION_SOURCE[mode]
        if mode == "full-scene" and pictures:
            # held best in tests: as a first-frame anchor the picture's layout overrode the video
            lead = "The target video is <Video 1> with its person replaced by <Subject 1>, the person shown in <Picture 1>."
            role += ", and its person is replaced by <Subject 1>. Only the appearance of <Subject 1> comes from <Picture 1>"
            keep = keep[:-1] + "; the person of <Video 1> is not rendered."
            follow = "<Subject 1> takes the place of the person in <Video 1> and moves as that person does, move for move; the setting and the camera are those of <Video 1>."
        return {"video_roles": [role], "video_keeps": [keep], "video_follows": [follow.format(who=who)],
                **({"lead": lead} if mode == "full-scene" and pictures else {})}
    kept = MOTION_KEEPS[mode]
    definitions = []
    if pictures:
        definitions.append("<Picture 1> is the first-frame anchor: the opening frame of the target video, with the subject, "
                           "setting and light of the target. The appearance of <Subject 1> comes from <Picture 1>"
                           + ("." if mode == "camera-only" else ", the position and motion from <Video 1>."))
    definitions.append(f"Legend of <Video 1>: {MOTION_LEGENDS[layer]}. These colours are labels only, not the look of the target video.")
    follow = ("The camera follows the path of <Video 1> frame by frame." if mode == "camera-only"
              else "The main object is where the main object of <Video 1> is in every frame and moves along the same path, and the camera "
                   "follows the path of <Video 1>." if mode == "object-motion" and not pictures
              else f"The position, scale and movement of {who.replace('The', 'the')} in every frame are those of the "
                   f"{'main object' if mode == 'object-motion' else 'figure'} in <Video 1>, and the camera follows the path of <Video 1>.")
    look = ", continuing the look of <Picture 1>" if pictures else ""
    return {"lead": "A photorealistic video that follows the structure of <Video 1> frame by frame"
                    + (" and has the look of <Picture 1>, with <Subject 1>." if pictures else "."),
            "video_roles": [f"the structural template of the target video: it supplies {kept} frame by frame"],
            "video_keeps": [f"fully_preserved structure - {kept} of <Video 1> are followed frame by frame; its look is not rendered."],
            "video_follows": [follow], "definitions": definitions,
            "avoid": f"The picture is photorealistic throughout{look}: no skeleton lines, no flat-coloured, depth-map or segmentation-map look."}


def freed(asked, motion: dict | None, pictures: int) -> bool:
    """Whether Impacts over a slow, measured reference leaves the video out: the model follows a reference's speed."""

    measured = (motion or {}).get("measured")
    return bool(motion and motion.get("free") and pictures > 0 and isinstance(measured, dict)
                and measured.get("motion") in ("slow", "moderate") and pace.choose(asked or motion.get("pace"), measured) == "impacts")


def paced_fields(fields: dict, motion: dict | None, has_video: bool, pictures: int) -> dict:
    """A reference clip's sound and motion made to agree (pace.py): returns ``{"fields", "pace"}``."""

    measured = (motion or {}).get("measured") if isinstance((motion or {}).get("measured"), dict) else None
    asked = fields.get("pace") or (motion or {}).get("pace")
    asked = asked if asked in pace.PACES else "auto"
    chosen = pace.choose(asked, measured)
    if asked == "auto" and not (chosen == "flow" and has_video and measured.get("motion") == "slow"
                                and pace.has_impacts(fields.get("soundscape") or "")):
        return {"fields": fields, "pace": ""}  # Automatic steps in only where hits were asked for over a slow reference
    if not (has_video or asked != "auto"):
        return {"fields": fields, "pace": ""}
    fields = dict(fields)
    try:
        many = int((motion or {}).get("people") or 1) > 1
    except (TypeError, ValueError):
        many = False
    action, soundscape = (fields.get("action") or "").strip(), (fields.get("soundscape") or "").strip()
    if chosen == "flow":
        fields["action"] = (action + " " + pace.SMOOTH).strip()
        # without music the clip is asked for near silence in so many words, or the model lays music of its own over it
        quiet = pace.calm(soundscape)
        named = (fields.get("music") or "N/A").strip().upper() != "N/A"
        fields["soundscape"] = quiet if named or quiet == pace.QUIET else "Near silence: " + quiet[0].lower() + quiet[1:]
        if asked == "flow" and measured and measured.get("sound") == "impacts":
            fields["video_sound"] = False  # Flow by choice: the reference's own hits are not reused
    else:
        who = "<Subject 1>" if pictures else ("The two" if many else "The subject")
        fields["action"] = (action + " " + (pace.VIGOROUS if many else pace.VIGOROUS_ONE).format(who=who)).strip()
        if not fields.get("video_sound") and not pace.has_impacts(soundscape):
            fields["soundscape"] = ("Sharp slaps and knocks of hands and forearms, one on every strike that lands. " + soundscape).strip()
    return {"fields": fields, "pace": chosen}


def handoff(fields: dict, count: int) -> tuple[dict, dict | None, bool]:
    """A Motion Previs hand-off applied to reference fields: (fields, motion, whether the video is left out)."""

    motion = fields.get("motion") if isinstance(fields.get("motion"), dict) else None
    free = len(fields.get("video_subjects") or []) <= 1 and freed(fields.get("pace"), motion, int(count))
    if free:  # the fight is the prompt's: no video goes in, and its camera move stays as words
        fields = {**fields, "video_subjects": [], "video_sound": False}
    if motion and (fields.get("video_subjects") or motion.get("layer") in (None, "", "none") or free):
        pictures = max(0, min(int(count), REFERENCE_LIMIT)) if fields.get("video_subjects") else 0
        made = motion_template(motion, pictures) if fields.get("video_subjects") and motion.get("layer") not in (None, "", "none") else {}
        fields = {**fields, **{key: value for key, value in made.items() if not fields.get(key)}}
        if not (fields.get("camera") or "").strip() and (motion.get("camera") or "").strip():
            fields["camera"] = motion["camera"].strip()
    return fields, motion, free
