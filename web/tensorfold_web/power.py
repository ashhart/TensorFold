"""Power, temperature and throttling, one source per platform, read without administrator rights.

- Mac: the SMC through IOKit. `PSTR` is the power the whole machine draws; the `Tp…` and `Tg…` probes are the CPU and
  GPU temperatures. `powermetrics` needs root, and `macmon` stops on chips it does not know yet (the M6 in 0.8.2).
- NVIDIA: `nvidia-smi`, the GPU's own power draw and temperature: part of the machine only, so energy is an estimate.
- AMD: `amd-smi`, the socket power of an APU (Strix Halo) and its temperature: also part of the machine.

Throttling, "did the machine slow itself down": the Mac's thermal state (serious or critical) from `NSProcessInfo`,
NVIDIA's active clock-event reasons (power cap, thermal, hardware slowdown), and amd-smi's throttle status when the
installed version reports it. None means not reported, which is not the same as "no".

A platform without a source says why; the Bench then runs without power figures instead of failing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import platform
import shutil
import struct
import subprocess
from dataclasses import dataclass


@dataclass
class Reading:
    watts: float | None = None
    cpu_c: float | None = None        # hottest CPU probe
    gpu_c: float | None = None        # hottest GPU probe
    throttled: bool | None = None     # None: this platform does not say
    reasons: tuple[str, ...] = ()     # why, in the platform's words


class Source:
    name = "none"
    scope = ""                        # what the watts cover: "whole machine", "GPU", "APU socket"
    whole_machine = False

    def read(self) -> Reading | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def describe(self) -> dict:
        return {"source": self.name, "scope": self.scope, "whole_machine": self.whole_machine}


# -- Mac: the SMC ---------------------------------------------------------------------------------------------------

class _Vers(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint8), ("minor", ctypes.c_uint8), ("build", ctypes.c_uint8),
                ("reserved", ctypes.c_uint8), ("release", ctypes.c_uint16)]


class _PLimit(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint16), ("length", ctypes.c_uint16), ("cpu", ctypes.c_uint32),
                ("gpu", ctypes.c_uint32), ("mem", ctypes.c_uint32)]


class _KeyInfo(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("type", ctypes.c_uint32), ("attributes", ctypes.c_uint8)]


class _KeyData(ctypes.Structure):
    """`SMCKeyData_t`, 80 bytes, as AppleSMC's user client takes it."""

    _fields_ = [("key", ctypes.c_uint32), ("vers", _Vers), ("plimit", _PLimit), ("info", _KeyInfo),
                ("result", ctypes.c_uint8), ("status", ctypes.c_uint8), ("data8", ctypes.c_uint8),
                ("data32", ctypes.c_uint32), ("bytes", ctypes.c_uint8 * 32)]


_READ_BYTES, _READ_INDEX, _READ_INFO = 5, 8, 9
_HANDLE_EVENT = 2


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode("ascii"))[0]


def _name(v: int) -> str:
    return struct.pack(">I", v).decode("ascii", "replace")


class MacThermal:
    """`NSProcessInfo.processInfo.thermalState` through the Objective-C runtime: 0 nominal to 3 critical."""

    STATES = ("nominal", "fair", "serious", "critical")

    def __init__(self) -> None:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        # objc_msgSend is called through a pointer cast to each method's exact signature, as arm64 requires
        get_obj = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        self._get_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(send)
        self._info = get_obj(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        self._sel = objc.sel_registerName(b"thermalState")
        if not self._info:
            raise OSError("no NSProcessInfo")

    def state(self) -> str | None:
        n = self._get_long(self._info, self._sel)
        return self.STATES[n] if 0 <= n < len(self.STATES) else None


class MacSMC(Source):
    name = "SMC"
    scope = "whole machine"
    whole_machine = True

    def __init__(self) -> None:
        iokit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit"))
        self._iokit = iokit
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceGetMatchingService.restype = ctypes.c_uint32
        iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        iokit.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        iokit.IOServiceClose.argtypes = [ctypes.c_uint32]
        iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
        iokit.IOConnectCallStructMethod.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t,
                                                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        libc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("c"))
        task_self = ctypes.c_uint32.in_dll(libc, "mach_task_self_").value
        service = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AppleSMC"))
        if not service:
            raise OSError("no AppleSMC service")
        conn = ctypes.c_uint32()
        kr = iokit.IOServiceOpen(service, task_self, 0, ctypes.byref(conn))
        iokit.IOObjectRelease(service)
        if kr != 0:
            raise OSError(f"AppleSMC refused the connection ({kr})")
        self._conn = conn.value
        self._info: dict[str, tuple[int, str]] = {}
        keys = self._all_keys()
        self.cpu_keys = [k for k in keys if k.startswith("Tp") and self._type(k) == "flt "]
        self.gpu_keys = [k for k in keys if k.startswith("Tg") and self._type(k) == "flt "]
        if self.value("PSTR") is None:
            raise OSError("this Mac's SMC has no system power key (PSTR)")
        try:
            self._thermal: MacThermal | None = MacThermal()
        except (OSError, AttributeError, TypeError):
            self._thermal = None

    def _call(self, inp: _KeyData) -> _KeyData | None:
        out = _KeyData()
        size = ctypes.c_size_t(ctypes.sizeof(out))
        kr = self._iokit.IOConnectCallStructMethod(self._conn, _HANDLE_EVENT, ctypes.byref(inp), ctypes.sizeof(inp),
                                                   ctypes.byref(out), ctypes.byref(size))
        return out if kr == 0 and out.result == 0 else None

    def _key_info(self, key: str) -> tuple[int, str] | None:
        if key not in self._info:
            inp = _KeyData(key=_fourcc(key), data8=_READ_INFO)
            out = self._call(inp)
            if out is None:
                return None
            self._info[key] = (out.info.size, _name(out.info.type))
        return self._info[key]

    def _type(self, key: str) -> str | None:
        info = self._key_info(key)
        return info[1] if info else None

    def _all_keys(self) -> list[str]:
        count = self._raw("#KEY")
        n = struct.unpack(">I", count)[0] if count and len(count) >= 4 else 0
        keys = []
        for i in range(n):
            out = self._call(_KeyData(data8=_READ_INDEX, data32=i))
            if out is not None:
                keys.append(_name(out.key))
        return keys

    def _raw(self, key: str) -> bytes | None:
        info = self._key_info(key)
        if info is None:
            return None
        inp = _KeyData(key=_fourcc(key), data8=_READ_BYTES)
        inp.info.size = info[0]
        out = self._call(inp)
        return bytes(out.bytes[: info[0]]) if out is not None else None

    def value(self, key: str) -> float | None:
        raw, kind = self._raw(key), self._type(key)
        if raw is None or kind is None:
            return None
        if kind == "flt " and len(raw) == 4:
            return struct.unpack("<f", raw)[0]
        if kind == "ui8 " and raw:
            return float(raw[0])
        if kind in ("ui16", "ui32") and raw:
            return float(int.from_bytes(raw, "big"))
        return None

    def _hottest(self, keys: list[str]) -> float | None:
        vals = [v for v in (self.value(k) for k in keys) if v is not None and 5.0 < v < 130.0]
        return max(vals) if vals else None

    def read(self) -> Reading:
        state = self._thermal.state() if self._thermal else None
        return Reading(watts=self.value("PSTR"), cpu_c=self._hottest(self.cpu_keys), gpu_c=self._hottest(self.gpu_keys),
                       throttled=None if state is None else state in ("serious", "critical"),
                       reasons=(f"thermal state {state}",) if state in ("fair", "serious", "critical") else ())

    def close(self) -> None:
        if self._conn:
            self._iokit.IOServiceClose(self._conn)
            self._conn = 0


# -- NVIDIA and AMD --------------------------------------------------------------------------------------------------

def _number(text: str) -> float | None:
    try:
        return float(str(text).strip().split()[0])
    except (ValueError, IndexError):
        return None


# clock-event reasons that mean the GPU slowed itself down; idle, application clocks and sync boost do not count
NVIDIA_SLOWDOWN = {0x4: "power cap", 0x8: "hardware slowdown", 0x20: "thermal (software)", 0x40: "thermal (hardware)",
                   0x80: "power brake"}


class NvidiaSmi(Source):
    name = "nvidia-smi"
    scope = "GPU"
    # the reasons' field was renamed in newer drivers; the first one this driver knows is kept
    REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active", None)

    def __init__(self) -> None:
        self._fields = list(self.REASON_FIELDS)

    def _query(self, reason: str | None) -> list[list[str]]:
        q = "power.draw,temperature.gpu" + (f",{reason}" if reason else "")
        done = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5, check=False)
        # a field this driver does not know is said so; a slow or busy nvidia-smi is not that
        self._unknown_field = "not a valid field" in (done.stdout + done.stderr).lower()
        return [[c.strip() for c in line.split(",")] for line in done.stdout.strip().splitlines() if "," in line]

    def read(self) -> Reading | None:
        try:
            rows = self._query(self._fields[0])
            # a field this driver does not know: the next one; an empty answer for another reason keeps the fields
            while not rows and self._fields[0] is not None and getattr(self, "_unknown_field", False):
                self._fields.pop(0)
                rows = self._query(self._fields[0])
        except (OSError, subprocess.SubprocessError):
            return None
        if not rows:
            return None
        watts = [w for w in (_number(r[0]) for r in rows) if w is not None]
        temps = [t for t in (_number(r[1]) for r in rows) if t is not None]
        throttled, reasons = None, set()
        if self._fields[0] is not None:
            masks = [int(r[2], 16) for r in rows if len(r) > 2 and r[2].lower().startswith("0x")]
            if masks:
                reasons = {name for m in masks for bit, name in NVIDIA_SLOWDOWN.items() if m & bit}
                throttled = bool(reasons)
        return Reading(watts=sum(watts) if watts else None, gpu_c=max(temps) if temps else None,
                       throttled=throttled, reasons=tuple(sorted(reasons)))


def _find(obj: object, names: tuple[str, ...]) -> float | None:
    """The first numeric value under any of ``names`` in amd-smi's JSON, whose layout varies between versions."""

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in names:
                n = v.get("value") if isinstance(v, dict) else v
                got = _number(n) if n not in (None, "N/A") else None
                if got is not None:
                    return got
            found = _find(v, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find(v, names)
            if found is not None:
                return found
    return None


def _find_text(obj: object, names: tuple[str, ...]) -> str | None:
    """The first value under any of ``names``, as text; "N/A" counts as missing."""

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in names and not isinstance(v, (dict, list)) and str(v) not in ("", "N/A"):
                return str(v)
            found = _find_text(v, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_text(v, names)
            if found is not None:
                return found
    return None


class AmdSmi(Source):
    name = "amd-smi"
    scope = "APU socket"

    def __init__(self) -> None:
        self._throttle = True              # dropped once amd-smi refuses `--throttle` (older versions)

    def _metric(self, *flags: str) -> object:
        out = subprocess.run(["amd-smi", "metric", *flags, "--json"], capture_output=True, text=True, timeout=5,
                             check=False).stdout
        return json.loads(out)

    def read(self) -> Reading | None:
        try:
            data = self._metric("--power", "--temperature")
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        throttled: bool | None = None
        if self._throttle:
            try:
                status = _find_text(self._metric("--throttle"), ("throttle_status",))
            except (OSError, subprocess.SubprocessError, ValueError):
                status = None
            if status is None:
                self._throttle = False
            else:
                throttled = status.lower() not in ("unthrottled", "false", "0", "no")
        return Reading(watts=_find(data, ("socket_power", "current_socket_power", "average_socket_power")),
                       gpu_c=_find(data, ("hotspot", "edge")), throttled=throttled,
                       reasons=("amd-smi throttle status",) if throttled else ())


# -- choosing ----------------------------------------------------------------------------------------------------------

def detect() -> tuple[Source | None, str]:
    """This machine's source, or None and the reason, said the same way on every platform."""

    if platform.system() == "Darwin":
        try:
            return MacSMC(), "SMC system power"
        except OSError as exc:
            return None, f"the SMC could not be read: {exc}"
    if shutil.which("nvidia-smi"):
        src = NvidiaSmi()
        r = src.read()
        if r and (r.watts is not None or r.gpu_c is not None):
            return src, "nvidia-smi"
        if not shutil.which("amd-smi"):
            return None, "nvidia-smi reports neither power nor temperature here"
    if shutil.which("amd-smi"):                       # also when an nvidia-smi present tells nothing (an AMD machine)
        src = AmdSmi()
        r = src.read()
        if r and (r.watts is not None or r.gpu_c is not None):
            return src, "amd-smi"
        return None, "amd-smi reports neither power nor temperature here"
    return None, "no power source on this machine (macOS SMC, nvidia-smi or amd-smi)"
