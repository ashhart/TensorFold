"""MiniMax H3 prompts from the create wizard's parts: the three-field form and the Ref2VA reference form."""

from __future__ import annotations

import re

import lengths
from motion_prompt import handoff, paced_fields
from settings import FPS, LEAD, REFERENCE_AUDIO_LIMIT, REFERENCE_LIMIT, REFERENCE_VIDEO_LIMIT


def lead(first: bool, last: bool, seconds: float = 5, frames: int | None = None) -> str:
    """The line that opens a keyframe prompt: which picture is fully referenced at which second of the clip."""

    end = f"{(frames or lengths.nearest(float(seconds))) / FPS:.2f}"
    if first and last:
        return LEAD[:-1] + f"; at {end} seconds into the target video, <Picture 2> (from [Shot 1]) is fully referenced."
    if last:
        return f"For the target video, at {end} seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    return LEAD if first else ""


# What a clip without spoken lines is told: the model takes no negative prompt (negation.py), so say what is there.
QUIET_PICTURE = "Mouths stay closed."
QUIET_SOUND = "Only ambient sound is heard."
QUIET_AFTER = "After the spoken lines only ambient sound is heard."
PEOPLE = re.compile(r"\b(man|men|woman|women|person|people|boy|boys|girl|girls|child|children|kid|kids|baby|he|she|they|him|her|his|them|their|"
                    r"someone|somebody|everyone|crowd|couple|friends?|students?|dancers?|singers?|actors?|actress|workers?|soldiers?|"
                    r"mother|father|son|daughter|brother|sister|husband|wife|face|faces|\w+ (?:is|are) \d+ years old)\b", re.IGNORECASE)


def quiet(fields: dict, *texts: str) -> str:
    """The sentence a shot without spoken lines ends on: QUIET_PICTURE where the shot has people, else nothing."""

    people = fields.get("people")
    if people is None:
        people = bool(PEOPLE.search(" ".join(text or "" for text in texts)))
    return " " + QUIET_PICTURE if people else ""


def said_quiet(sound: str) -> bool:
    """Whether a soundscape already says that there is no speech, in the old words or the new."""

    low = sound.lower()
    return "no speech" in low or "only ambient sound" in low


def compose_references(fields: dict, count: int) -> str:
    """The Ref2VA prompt form: who each reference is, what is kept of it, then the shot and its sound (README)."""

    fields, motion, free = handoff(fields, count)

    videos = [(text or "").strip() for text in (fields.get("video_subjects") or [])][:REFERENCE_VIDEO_LIMIT]
    audios = [(text or "").strip() for text in (fields.get("audio_notes") or [])][:REFERENCE_AUDIO_LIMIT]
    count = max(0 if videos else 1, min(int(count), REFERENCE_LIMIT))
    paced = paced_fields(fields, motion, bool(videos) or free, max(0, min(int(count), REFERENCE_LIMIT)))
    fields, pacing = paced["fields"], paced["pace"]
    names = [(text or "").strip() for text in (fields.get("subjects") or [])][:count]
    names += [""] * (count - len(names))
    subjects = [f"<Subject {i}> is {name or 'the main subject'} shown in <Picture {i}>." for i, name in enumerate(names, start=1)]
    kept = [f"<Subject {i}> (appears in [Shot 1]): fully_preserved - the identity, face, hair, clothing and colours "
            f"shown in <Picture {i}> are kept." for i in range(1, count + 1)]
    people, sounds, follow, tracks = count, 0, [], []
    # the speaker's voice in the user's words, beside an audio reference too: the model follows a described voice
    described = next(((line.get("voice") or "").strip() for line in fields.get("dialogue") or []
                      if (line.get("text") or "").strip() and (line.get("voice") or "").strip()), "")
    roles = [(text or "").strip() for text in (fields.get("video_roles") or [])]
    keeps = [(text or "").strip() for text in (fields.get("video_keeps") or [])]
    follows = [(text or "").strip() for text in (fields.get("video_follows") or [])]
    for k, shown in enumerate(videos, start=1):
        role, keep = (roles[k - 1] if k <= len(roles) else ""), (keeps[k - 1] if k <= len(keeps) else "")
        if role:  # the caller's own account of this video
            subjects.append(f"<Video {k}> is {role.rstrip('.')}.")
            kept.append(f"<Video {k}>: " + ("" if keep.startswith("fully_preserved") else "reference - ")
                        + (keep or f"the structure and movement of <Video {k}> are followed."))
            follow.append((follows[k - 1] if k <= len(follows) else "") or f"The layout, movement and camera path follow <Video {k}>.")
        elif shown or not people:  # a video that shows someone: a subject of its own, as the model card writes it
            people += 1
            subjects.append(f"<Subject {people}> is {shown or 'the main subject'} in <Video {k}>.")
            subjects.append(f"<Video {k}> is the reference video for the appearance and the motion of <Subject {people}>.")
            kept.append(f"<Subject {people}> (appears in [Shot 1]): fully_preserved - the identity, face, hair, "
                        f"clothing and colours shown in <Video {k}> are kept.")
            kept.append(f"<Video {k}>: reference - the motion and timing of <Subject {people}> in <Video {k}> are followed.")
            follow.append(f"<Subject {people}> moves as in <Video {k}>.")
        else:
            subjects.append(f"<Video {k}> is the motion reference for the target video.")
            kept.append(f"<Video {k}>: reference - the motion, timing and camera movement of <Video {k}> are followed.")
            follow.append(f"The motion and camera movement follow <Video {k}>.")
        if fields.get("video_sound"):
            sounds += 1
            subjects.append(f"<Audio {sounds}> is the synchronized audio track of <Video {k}>.")
            kept.append(f"<Audio {sounds}>: reference - the sound of <Audio {sounds}> is reused in the target video.")
            tracks.append(f"<Audio {sounds}>")
    subjects += [line.strip() for line in (fields.get("definitions") or []) if line and line.strip()]
    voices = []
    for note in audios:
        sounds += 1
        subjects.append(f"<Audio {sounds}> is " + (note.rstrip(".") if note else "the voice timbre reference for <Subject 1>'s voice"
                                                   + (f", containing {described}" if described else "")) + ".")
        kept.append(f"<Audio {sounds}>: reference - the target audio references the voice timbre from <Audio {sounds}>"
                    + " for the spoken lines." if described or not note
                    else f"<Audio {sounds}>: reference - the target audio references the sound of <Audio {sounds}>.")
        voices.append(f"<Audio {sounds}>")
    scene = (fields.get("scene") or "").strip()
    shot = " ".join(part.strip() for part in (scene, fields.get("camera", ""), fields.get("action", ""), *follow)
                    if part and part.strip())
    spoken = []
    for index, line in enumerate(fields.get("dialogue") or [], start=1):
        text = (line.get("text") or "").strip()
        if not text:
            continue
        voice = (line.get("voice") or "a natural voice").strip()
        verb = "sings" if line.get("sung") else "speaks"
        language = (line.get("language") or "English").strip()
        spoken.append(f"<Subject {int(line.get('subject') or 1)}> (S{index}), in {voice}, {verb}: <d>[{language}] {text}</d>")
    if spoken:
        if voices:
            shot += " The speaker's voice timbre references the delivery from " + " and ".join(voices) + "."
        # a clip longer than its lines: without this the model may keep the mouth moving and invent words
        shot += (" " + " ".join(spoken) + " Mouth movements follow the words. After the last word the speaker stops"
                 " talking, closes the mouth and stays silent until the end of the clip.")
    elif tracks:  # a video's own sound is a reference: it may hold speech, so nothing here says that nobody speaks
        shot += " The sound of " + " and ".join(tracks) + " plays through the clip."
    else:
        # a reference picture or a video with a subject is almost always a person, unless the fields say otherwise
        shot += quiet({"people": True if fields.get("people") is None and people else fields.get("people")}, scene, fields.get("action", ""))
    if (fields.get("avoid") or "").strip():
        shot += " " + fields["avoid"].strip()
    sound = (fields.get("soundscape") or ("The sound of " + " and ".join(tracks) + ", reused." if tracks
                                          else "Natural ambient sound for the scene.")).strip()
    sound += " " + QUIET_AFTER if spoken else "" if tracks else " " + QUIET_SOUND
    if voices:
        sound += (" The spoken lines reference the voice timbre of " if spoken else " The sound references ") + " and ".join(voices) + "."
    music = (fields.get("music") or "N/A").strip()
    if music.upper() != "N/A":
        sound += f" Music: {music}."
    who = ", ".join(f"<Subject {i}>" for i in range(1, people + 1))
    kinds = " + ".join(kind for kind, used in (("subject reference", count), ("video reference", videos),
                                               ("audio reference", sounds)) if used)
    return ("subject_definitions:\n" + "\n".join(subjects)
            + f"\n\nsummary:\n[{kinds}] "
            + ((fields.get("lead") or "").strip() + " " if (fields.get("lead") or "").strip()
               else f"A new video with {who}, keeping how each looks in its " + ("reference. " if videos else "picture. ") if who
               else "A new video that follows its video reference. ")
            + (scene or "A new scene.")
            + "\n\nretention_analysis:\n" + "\n".join(kept)
            # the sound under its own heading: on the end of the shot the model reads it as something to say
            + f"\n\ndetailed_description:\n[Shot 1] {shot.strip()}\n\noverall_soundscape:\n{sound}")


def compose(fields: dict, first_frame: bool, last_frame: bool = False, seconds: float = 5, frames: int | None = None) -> str:
    """MiniMax H3's three-field prompt from the wizard's parts."""

    shot = " ".join(part.strip() for part in (fields.get("scene", ""), fields.get("camera", ""),
                                               fields.get("action", "")) if part and part.strip())
    spoken = []
    for index, line in enumerate(fields.get("dialogue") or [], start=1):
        text = (line.get("text") or "").strip()
        if not text:
            continue
        voice = (line.get("voice") or "a natural voice").strip()
        verb = "sings" if line.get("sung") else "says"
        language = (line.get("language") or "English").strip()
        spoken.append(f"{(line.get('speaker') or '').strip() or 'The speaker'}, in {voice} (S{index}), {verb}: <d>[{language}] {text}</d>")
    if spoken:
        shot += (" " + " ".join(spoken) + " Mouth movements follow the words. After the last word the speaker stops"
                 " talking, closes the mouth and stays silent until the end of the clip.")
    else:
        shot += quiet(fields, fields.get("scene", ""), fields.get("action", ""))
    sound = (fields.get("soundscape") or "Natural ambient sound for the scene.").strip()
    if not said_quiet(sound):
        sound += " " + QUIET_AFTER if spoken else " " + QUIET_SOUND
    music = (fields.get("music") or "N/A").strip()
    body = (f"integrated_multimodal_description: [Shot 1] {shot.strip()}\n\n"
            f"overall_soundscape: {sound}\n\nnon_diegetic_music: {music}")
    opening = lead(first_frame, last_frame, seconds, frames)
    return f"{opening}\n\n{body}" if opening else body
