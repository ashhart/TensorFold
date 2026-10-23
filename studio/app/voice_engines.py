"""TensorFold's two Drama 3 engines: MLX on a Mac (``Mlx``) and CUDA on a DGX Spark (``Cuda``)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from voice_machine import comfy_up, mem_available_gb
from voice_served import LIVE, Served, forget_live, join_wavs
from voice_text import checked_by_engine, dialogue, expected_seconds, one_speaker, pieces, spoken


MEMORY_GB = 64  # a Mac with less does not run it: the model peaks at about 52 GB


class Mlx(Served):
    """TensorFold's MLX Drama 3 in its own venv: a ``--serve`` server (92c09a8e on), else a process a take."""

    name, label = "mlx", "TensorFold MLX · Apple Silicon"
    fps, load_s = 75.0, 5.5  # measured on an M5 Ultra: 70 to 73 frames a second streamed, 83 not; 5.3 s to ready
    NEED_GB = 58.0  # free memory wanted before it loads: about 50 GB with Whisper, and the guard's 8 GB beside it

    def __init__(self) -> None:
        super().__init__()
        self.venv = Path(os.environ.get("DRAMA3_VENV", Path.home() / "drama3-venv")).expanduser()
        self.model = Path(os.environ.get("DRAMA3_MODEL", Path.home() / "models/drama-3-mlx")).expanduser()
        self.known: tuple[float, str] | None = None  # (when, why not) of the last look: a look runs a process
        # --stream (a growing WAV) and its own --verify end-of-text check: drama3-mlx 4a8a982f or later
        self.streams = False
        self.serves = False  # --serve: drama3-mlx 92c09a8e or later
        self.hears = False  # --serve hears a reference left without its text: drama3-mlx 8f3d4b33 or later

    def serving(self) -> bool:
        return self.serves and self.check() == ""

    @property
    def checks(self) -> bool:
        """Whether the engine checks its own words (drama3-mlx 4a8a982f on): the app then does not."""
        return self.streams

    def argv(self) -> list[str]:
        return [str(self.python()), "-I", "-m", "tensorfold.drama3", "--model", str(self.model), "--serve"]

    def python(self) -> Path:
        return self.venv / "bin" / "python"

    def check(self) -> str:
        """Why Drama 3 cannot run here ("" when it can): looked at once a minute at most, never longer than 20 s."""

        if self.loaded():
            return ""
        if self.known and (not self.known[1] or time.time() - self.known[0] < 60):
            return self.known[1]
        why = self.look()
        self.known = (time.time(), why)
        return why

    def look(self) -> str:
        if sys.platform != "darwin":
            return "The MLX backend runs on a Mac (Apple Silicon)."
        if not self.python().is_file():
            return f"No Drama 3 environment at {self.venv} (README, Voice: TensorFold's drama3-mlx branch in its own venv)."
        if not (self.model / "config.json").is_file() or not (self.model / "codec.safetensors").is_file():
            return (f"No converted Drama 3 model at {self.model} (python -m tensorfold.families.drama3.convert "
                    "~/models/drama-3 ~/models/drama-3-mlx).")
        try:
            memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip() or 0)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            memory = 0
        if memory and memory < MEMORY_GB * 1024 ** 3:
            return f"This Mac has {memory / 1024 ** 3:.0f} GB of memory; Drama 3 needs about 52 GB free, so {MEMORY_GB} GB or more."
        try:
            done = subprocess.run([str(self.python()), "-I", "-c", "import importlib.util as u; s = u.find_spec('tensorfold.drama3'); "
                                   "print(bool(s), bool(u.find_spec('tensorfold.families.drama3.stream')), "
                                   "bool(s and s.origin and '\"--serve\"' in open(s.origin).read()), "
                                   "bool(s and s.origin and 'ref_text_optional' in open(s.origin).read()))"],
                                  capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"The Drama 3 environment did not answer: {error}"
        found = done.stdout.split()
        if not found or found[0] != "True":
            return f"The environment at {self.venv} has no tensorfold.drama3 (install TensorFold's drama3-mlx branch into it)."
        self.streams = found[1:2] == ["True"]
        self.serves = found[2:3] == ["True"]
        self.hears = found[3:4] == ["True"]
        return ""

    def command(self, text_file: Path, out: Path, stats: Path, reference: tuple[Path, str] | None, seed: int, sampling: dict,
                live: bool = False) -> list[str]:
        command = [str(self.python()), "-I", "-m", "tensorfold.drama3", "--model", str(self.model), "--text-file", str(text_file),
                   "--out", str(out), "--stats", str(stats), "--seed", str(seed), "--temperature", f"{sampling['temperature']:g}",
                   "--top-p", f"{sampling['top_p']:g}", "--top-k", str(int(sampling["top_k"]))]
        if reference:
            command += ["--ref-audio", str(reference[0]), "--ref-text", reference[1]]
        if self.streams and live:  # the WAV at --out grows as it is spoken (its sizes are set when done): /live reads it
            command += ["--stream"]
        return command

    def make(self, job: dict, text: str, wav: Path, reference: tuple[Path, str] | None, seed: int, sampling: dict,
             live: bool = True) -> dict | None:
        """One take, one request; a part short while played is asked again unstreamed; None when cancelled."""

        folder = wav.with_suffix(".live")
        folder.mkdir(exist_ok=True)
        pcm = folder / "0.pcm"
        if live:
            LIVE[job["id"]] = {"parts": [pcm], "done": False, "finished": 0, "gap_s": 0}
        request = {"text": dialogue(text), "wav": str(wav), "seed": seed, "temperature": sampling["temperature"],
                   "top_k": int(sampling["top_k"]), "top_p": sampling["top_p"], "id": job["id"]}
        if reference:
            request.update(ref_audio=str(reference[0]), ref_text=reference[1])
        seconds = 300 + expected_seconds(text) * 4  # frames, the check and two redraws a part, with room
        with self.mutex:
            began = time.time()
            try:
                if not self.start(job):
                    return None
                loaded = time.time()
                job["progress"] = {"stage": "Drama 3 voice", "at": 0, "of": 1}
                answer = self.ask(job, {**request, "pcm": str(pcm), "stream": True} if live else {**request, "stream": False}, seconds)
                if answer is None:
                    return None
                retaken = None
                if live and any(isinstance(part, dict) and part.get("action") == "redraw" for part in answer.get("verify") or []):
                    # played audio cannot be redrawn: the take is asked again without streaming
                    retaken = {"played": checked_by_engine(answer), "seconds": answer.get("audio_s")}
                    job["log"] += "[tensorfold] a part was short while playing: the take is made again without playing, so it can be redrawn\n"
                    job["progress"] = {"stage": "Drama 3 voice, again without playing", "at": 0, "of": 1}
                    again = self.ask(job, {**request, "stream": False}, seconds)
                    if again is None:
                        return None
                    answer = again
            finally:
                self.last_used = time.time()
                if job["id"] in LIVE:
                    LIVE[job["id"]].update(finished=1, done=True)
                threading.Timer(60, forget_live, args=(job["id"], folder)).start()
        chunks = answer.get("chunks")
        return {"frames": answer.get("frames"), "audio_s": answer.get("audio_s"), "frames_per_s": answer.get("fps"),
                "rtf": answer.get("rtf"), "total_s": round(time.time() - began, 1),
                "engine_s": round(float(answer.get("total_ms") or 0) / 1000, 2), "first_audio_ms": answer.get("first_audio_ms"),
                "encode_ms": answer.get("encode_ms"), "load_s": job.get("load_s"), "peak_memory_gb": answer.get("peak_memory_gb"),
                "chunks": len(chunks) if isinstance(chunks, list) else chunks, "finish": answer.get("finish"),
                "verify": answer.get("verify"), "stream": bool(answer.get("stream")), "retaken": retaken,
                "ref_texts": answer.get("ref_texts"), "ref_asr_ms": answer.get("ref_asr_ms"),
                "made_s": round(time.time() - loaded, 2)}


class Cuda(Served):
    """TensorFold's CUDA Drama 3 (``drama3-generate --serve 1``, Zig + CUDA) on a DGX Spark; long text in ``pieces``."""

    name, label = "cuda", "TensorFold CUDA · DGX Spark"
    fps, load_s = 30.0, 18.0  # measured on a GB10: 29 to 31 frames a second sampled, 17 to 18 s to load from NFS
    NEED_GB = 50.0  # free memory wanted before it loads: 41.6 GB at the peak, and the guard's 8 GB beside it
    CHUNK_WORDS = 60  # a request's text at most (about 27 s of speech, ~600 frames, well inside max_ctx 4096)
    GAP_S = 0.15  # the silence put between two parts of a long take

    def __init__(self) -> None:
        super().__init__()
        prefix = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio")).expanduser()
        here = prefix / "drama3"
        self.bin = Path(os.environ.get("DRAMA3_CUDA_BIN", here / "bin" / "drama3-generate")).expanduser()
        self.lib = Path(os.environ.get("DRAMA3_CUDA_LIB", self.bin.parent.parent / "lib" / "libtensorfold_drama3.so")).expanduser()
        self.model = Path(os.environ.get("DRAMA3_CUDA_MODEL", Path.home() / "models" / "drama-3")).expanduser()
        self.codec = Path(os.environ.get("DRAMA3_CUDA_CODEC", here / "codec-v1.safetensors")).expanduser()

    checks = False  # no end-of-text check of its own: the app's (``redrawn``)

    def serving(self) -> bool:
        return True

    def environment(self) -> dict:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(self.lib.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        return env

    # -- availability --

    def check(self) -> str:
        """Why Drama 3 cannot run here ("" when it can): files looked at, then memory; at most every 10 s."""

        if self.process is not None and self.process.poll() is None:
            return ""
        if self.known and time.time() - self.known[0] < 10:
            return self.known[1]
        why = self.look()
        self.known = (time.time(), why)
        return why

    def look(self) -> str:
        if not sys.platform.startswith("linux"):
            return "The CUDA backend runs on a DGX Spark (Linux, CUDA)."
        if not os.access(self.bin, os.X_OK):
            return f"No Drama 3 engine at {self.bin} (README, Voice: build TensorFold's drama3-cuda branch; DRAMA3_CUDA_BIN names another)."
        if not self.lib.is_file():
            return f"No libtensorfold_drama3.so at {self.lib} (the same build makes it; DRAMA3_CUDA_LIB names another)."
        if not (self.model / "config.json").is_file() or not (self.model / "model.safetensors.index.json").is_file():
            return f"No Drama 3 checkpoint at {self.model} (hf download fishaudio/drama-3; DRAMA3_CUDA_MODEL names another)."
        if not self.codec.is_file():
            return (f"No converted codec at {self.codec} (once: codec_convert.py {self.model} {self.codec}; "
                    "DRAMA3_CUDA_CODEC names another).")
        free = mem_available_gb()
        if 0 <= free < self.NEED_GB and not comfy_up():
            return (f"Drama 3 needs about {self.NEED_GB:.0f} GB of free memory and this Spark has {free:.0f} GB available now "
                    "(something outside the studio holds the rest).")
        return ""

    def argv(self) -> list[str]:
        return [str(self.bin), "--model", str(self.model), "--codec", str(self.codec), "--serve", "1"]

    # -- a take --

    def make(self, job: dict, text: str, wav: Path, reference: tuple[Path, str] | None, seed: int, sampling: dict,
             live: bool = True) -> dict | None:
        """One take in parts, later parts of a one-voice take cloning the first; its stats, or None when cancelled."""

        parts = pieces(text, self.CHUNK_WORDS)
        folder = wav.with_suffix(".live")
        if live:  # played while it is made: the engine writes each part's raw PCM there as it goes (/live)
            folder.mkdir(exist_ok=True)
            LIVE[job["id"]] = {"parts": [folder / f"{i}.pcm" for i in range(len(parts))], "done": False, "finished": 0, "gap_s": self.GAP_S}
        with self.mutex:
            began = time.time()
            if not self.start(job):
                return None
            loaded = time.time()
            answers, files = [], []
            try:
                for index, part in enumerate(parts):
                    job["progress"] = {"stage": "Drama 3 voice" + (f", part {index + 1} of {len(parts)}" if len(parts) > 1 else ""),
                                       "at": index, "of": len(parts)}
                    out = wav if len(parts) == 1 else wav.with_name(f"{wav.stem}.part{index}.wav")
                    request = {"text": dialogue(part), "wav": str(out), "seed": seed + index,
                               "temperature": sampling["temperature"], "top_k": int(sampling["top_k"]),
                               "top_p": sampling["top_p"], "max_frames": 1500}
                    if live:
                        request["pcm"] = str(LIVE[job["id"]]["parts"][index])
                    voice_of = reference
                    if not voice_of and index and files and one_speaker(text):
                        voice_of = (files[0], spoken(parts[0]))
                    if voice_of:
                        request.update(ref_audio=str(voice_of[0]), ref_text=voice_of[1])
                    answer = self.ask(job, request, 120 + 1500 / 10)
                    if answer is None:
                        return None
                    answers.append(answer)
                    files.append(out)
                    if live:
                        LIVE[job["id"]]["finished"] = index + 1
                    self.last_used = time.time()
            finally:
                self.last_used = time.time()
                if live:
                    LIVE[job["id"]]["done"] = True
                    # the raw parts go a minute later: a page may still be playing the end of the live take
                    threading.Timer(60, forget_live, args=(job["id"], folder)).start()
        if len(files) > 1:
            join_wavs(files, wav, self.GAP_S)
            for one in files:
                one.unlink(missing_ok=True)
        frames = sum(int(a.get("frames") or 0) for a in answers)
        audio = sum(float(a.get("audio_s") or 0) for a in answers) + self.GAP_S * (len(answers) - 1)
        decoding = sum(int(a.get("frames") or 0) / float(a["fps"]) for a in answers if a.get("fps"))
        made = time.time() - loaded
        return {"frames": frames, "audio_s": round(audio, 3), "frames_per_s": round(frames / decoding, 1) if decoding else None,
                "rtf": round(made / audio, 2) if audio else None, "total_s": round(time.time() - began, 1),
                "engine_s": round(sum(float(a.get("total_ms") or 0) for a in answers) / 1000, 2),
                "first_audio_ms": answers[0].get("first_audio_ms") if answers else None,
                "encode_ms": answers[0].get("encode_ms") if answers else None,
                "load_s": job.get("load_s"), "device_gb": self.device_gb,
                "chunks": len(answers), "finish": ["stop" if a.get("eos") else "length" for a in answers], "stream": live}
