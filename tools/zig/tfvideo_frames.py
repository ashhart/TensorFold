"""libtfvideo's RGB frames against PyAV's (read_video_pyav) or another libtfvideo build, byte for byte, per clip."""

import argparse
import ctypes
import hashlib
import json
import sys
import time
from pathlib import Path

TF_DECODE, TF_SHORT = -5, -6


class Info(ctypes.Structure):
    _fields_ = [("frames", ctypes.c_int64), ("rate_num", ctypes.c_int), ("rate_den", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int), ("counted", ctypes.c_int)]


FRAME_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int)


class Library:
    """One libtfvideo build, loaded by path so two builds can share the process."""

    def __init__(self, path: Path):
        self.name = str(path)
        self.lib = ctypes.CDLL(str(path.resolve()), mode=ctypes.RTLD_LOCAL)
        self.lib.tf_video_open.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.POINTER(Info)]
        self.lib.tf_video_frames.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, FRAME_FN,
                                             ctypes.c_void_p]
        self.lib.tf_video_close.argtypes = [ctypes.c_void_p]
        self.lib.tf_video_version.restype = ctypes.c_char_p
        self.version = self.lib.tf_video_version().decode()

    def open(self, data: bytes) -> tuple[int, Info]:
        """tf_video_open's return code and the stream's facts."""
        handle, info = ctypes.c_void_p(), Info()
        rc = self.lib.tf_video_open(data, len(data), ctypes.byref(handle), ctypes.byref(info))
        if rc == 0:
            self.lib.tf_video_close(handle)
        return rc, info

    def frames(self, data: bytes, indices: list[int], hashed: bool = True) -> tuple[list, float, int]:
        """The frames delivered (hashed, or index only), the seconds tf_video_frames took and its return code."""
        handle, info = ctypes.c_void_p(), Info()
        rc = self.lib.tf_video_open(data, len(data), ctypes.byref(handle), ctypes.byref(info))
        if rc != 0:
            return [], 0.0, rc
        out: list[tuple[int, str]] = []

        def take(_ctx, _k, index, rgb, width, height, stride):
            if not hashed:  # timing: the bytes are left alone, so the decoder's threads never wait on Python
                out.append((index, ""))
                return 0
            raw = ctypes.string_at(rgb, stride * (height - 1) + width * 3)
            out.append((index, hashlib.sha256(b"".join(raw[r * stride:r * stride + width * 3]
                                                        for r in range(height))).hexdigest()))
            return 0

        callback = FRAME_FN(take)
        start = time.perf_counter()
        rc = self.lib.tf_video_frames(handle, (ctypes.c_int * len(indices))(*indices), len(indices), callback, None)
        seconds = time.perf_counter() - start
        self.lib.tf_video_close(handle)
        return out, seconds, rc


class PyAV:
    """transformers' read_video_pyav: decode from the start, keep the wanted indices, to_ndarray(format="rgb24")."""

    def __init__(self):
        import av
        self.av, self.name, self.version = av, f"PyAV {av.__version__}", str(av.ffmpeg_version_info)

    def open(self, data: bytes) -> tuple[int, None]:
        """0 when PyAV opens the container with a video stream, -2 (TF_NO_STREAM) without one, else -1 (TF_INVALID)."""
        import io
        try:
            with self.av.open(io.BytesIO(data)) as c:
                return (0 if c.streams.video else -2), None
        except self.av.error.FFmpegError:
            return -1, None

    def frames(self, data: bytes, indices: list[int], hashed: bool = True) -> tuple[list, float, int]:
        """As Library.frames; an FFmpeg error counts as TF_DECODE and an early end as TF_SHORT."""
        import io
        want, out, rc = set(indices), [], 0
        start = time.perf_counter()
        try:
            with self.av.open(io.BytesIO(data)) as c:
                c.seek(0)
                for i, frame in enumerate(c.decode(video=0)):
                    if i > indices[-1]:
                        break
                    if i in want:
                        rgb = frame.to_ndarray(format="rgb24")
                        out.append((i, hashlib.sha256(rgb.tobytes()).hexdigest() if hashed else ""))
        except self.av.error.FFmpegError:
            rc = TF_DECODE
        seconds = time.perf_counter() - start
        return out, seconds, rc or (len(out) if len(out) == len(indices) else TF_SHORT)


def index_sets(info: Info) -> dict[str, list[int]]:
    """Every frame, GLM's 2 frames a second, the first three (an early stop) and the last (the end-of-stream drain)."""
    n, rate = int(info.frames), info.rate_num / info.rate_den if info.rate_den else 0.0
    two = sorted({min(n - 1, round(i * rate / 2)) for i in range(int(n / rate * 2) + 1)}) if rate else [0]
    return {"all": list(range(n)), "2fps": two, "first3": list(range(min(3, n))), "last": [n - 1]}


def best(source, data: bytes, indices: list[int], repeat: int, hashed: bool) -> tuple[list, float, int]:
    frames, seconds, rc = source.frames(data, indices, hashed)
    for _ in range(repeat - 1):
        seconds = min(seconds, source.frames(data, indices, hashed)[1])
    return frames, seconds, rc


def compare(lib, ref, clip: Path, repeat: int, rows: dict) -> int:
    """One clip's sets against the reference (or timed alone when ref is None); returns the sets that differ."""
    data = clip.read_bytes()
    rc, info = lib.open(data)
    ref_rc, ref_info = ref.open(data) if ref else (rc, info)
    if rc != 0 or ref_rc != 0 or info.frames <= 0:  # a damaged clip may not open, or may count no frames
        rows["open"] = {"test_rc": rc, "reference_rc": ref_rc, "frames": info.frames}
        theirs = (ref_info or info).frames  # PyAV counts nothing here: libtfvideo's count stands for both
        ok = ref is not None and rc == ref_rc and (info.frames > 0) == (theirs > 0)
        print(f"{'ok  ' if ok else 'FAIL'} {clip} open: test rc {rc}, reference rc {ref_rc}, {info.frames} frames")
        return int(not ok)
    failures = 0
    for name, indices in index_sets(info).items():
        mine, mine_s, mine_rc = best(lib, data, indices, repeat, ref is not None)
        theirs, theirs_s, theirs_rc = best(ref, data, indices, repeat, True) if ref else (mine, 0.0, mine_rc)
        same = sum(a == b for a, b in zip(mine, theirs))
        ok = mine == theirs and mine_rc == theirs_rc and (ref is not None or mine_rc == len(indices))
        failures += not ok
        rows[name] = {"indices": indices, "test": mine, "reference": theirs if ref else None, "test_rc": mine_rc,
                      "reference_rc": theirs_rc, "test_s": mine_s, "reference_s": theirs_s}
        codes = f", rc {mine_rc}/{theirs_rc}" if (mine_rc, theirs_rc) != (len(indices), len(indices)) else ""
        timing = f"test {mine_s:.3f} s" + (f", reference {theirs_s:.3f} s" if ref else "")
        what = "frames equal" if ref else "frames delivered"
        print(f"{'ok  ' if ok else 'FAIL'} {clip} {name}: {same}/{len(indices)} {what}{codes}, {timing}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True, type=Path, help="the libtfvideo build under test")
    ap.add_argument("--against", type=Path, help="another libtfvideo build; PyAV when omitted")
    ap.add_argument("--time-only", action="store_true",
                    help="no reference and no hashing: time tf_video_frames alone (its callback leaves the bytes)")
    ap.add_argument("--repeat", type=int, default=1, help="decodes per set; the fastest time is reported")
    ap.add_argument("--json", type=Path, help="write every frame's hash, return code and each set's time here")
    ap.add_argument("clips", nargs="+", type=Path)
    args = ap.parse_args()
    lib = Library(args.lib)
    ref = None if args.time_only else Library(args.against) if args.against else PyAV()
    print(f"test {lib.name} (FFmpeg {lib.version}); reference "
          + (f"{ref.name} (FFmpeg {ref.version})" if ref else "none (--time-only)"))
    failures, report = 0, {"lib": lib.name, "reference": ref.name if ref else None, "clips": {}}
    for clip in args.clips:
        rows = report["clips"][str(clip)] = {}
        try:
            failures += compare(lib, ref, clip, args.repeat, rows)
        except Exception as e:  # one unreadable clip is a failed clip, not the end of the run
            rows["error"] = repr(e)
            failures += 1
            print(f"FAIL {clip}: {e!r}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    print(f"{failures} set(s) differ" if failures else "every frame equal")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
