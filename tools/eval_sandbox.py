"""Run a model's Python program in a fresh directory under a wall timeout, rlimits and an OS sandbox."""
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

try:
    import resource
except ImportError:  # Windows has no rlimits, so it gets no sandbox
    resource = None

POLL_S = 0.1  # how often the parent checks the clock and the program's memory
FILE_BYTES = 16 << 20  # the largest file the program may write, its captured output included
NO_OUTPUT = "[no output; print the values you need]"


class SandboxUnavailable(RuntimeError):
    """No sandbox is available on this host and the caller did not accept running without one."""


def symlinks(path: str, depth: int = 0) -> set[str]:
    """Each symlink met while resolving path, so a profile that denies home reads still lets exec follow them."""
    found, current = set(), os.sep
    for part in os.path.abspath(path).split(os.sep)[1:]:
        current = os.path.join(current, part)
        if os.path.islink(current) and depth < 20:
            found.add(current)
            found |= symlinks(os.path.join(os.path.dirname(current), os.readlink(current)), depth + 1)
            current = os.path.realpath(current)
    return found


def mac_profile(work: str) -> str:
    """A sandbox-exec profile: no network, no writes outside work, no reads in the home folder but Python's own."""
    home = os.path.realpath(os.path.expanduser("~"))
    python = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, os.path.dirname(sys.executable)}
    reads = [f"(subpath {json.dumps(p)})" for p in sorted({os.path.realpath(p) for p in python} | {work})]
    reads += [f"(literal {json.dumps(p)})" for p in sorted(symlinks(sys.executable))]
    rules = ["(version 1)", "(allow default)", "(deny network*)", "(deny file-write*)",
             f"(allow file-write* (subpath {json.dumps(work)}) (literal \"/dev/null\"))"]
    if home != "/":
        rules += [f"(deny file-read* (subpath {json.dumps(home)}))", "(allow file-read* " + " ".join(reads) + ")"]
    return "\n".join(rules)


def unshare_flags() -> list[str] | None:
    """The unshare command that gives a program its own empty network namespace here, or None."""
    exe = shutil.which("unshare")
    if exe is None:
        return None
    for flags in (["-n"], ["-r", "-n"]):
        try:
            probe = subprocess.run([exe, *flags, sys.executable, "-c", ""], capture_output=True, timeout=10,
                                   check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return [exe, *flags]
    return None


def group_rss_kib(pgid: int) -> int:
    """Resident memory of every process in the group, in KiB, from ps."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, timeout=5,
                             check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    total = 0
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == str(pgid) and fields[1].isdigit():
            total += int(fields[1])
    return total


def cut(text: str, limit: int) -> str:
    """Keep the head and the tail of text that is longer than limit, where a traceback ends."""
    if len(text) <= limit:
        return text
    tail = limit // 3
    head = limit - tail
    return f"{text[:head]}\n[... {len(text) - head - tail} characters cut ...]\n{text[-tail:]}"


def signal_name(code: int) -> str:
    try:
        return signal.Signals(-code).name
    except ValueError:
        return f"signal {-code}"


class Sandbox:
    """Runs one program at a time, each in a new temporary directory, under the limits it was built with."""

    def __init__(self, kind: str, prefix: list[str], timeout_s: float = 30.0, memory_mb: int = 2048,
                 output_chars: int = 4000):
        self.kind, self.prefix = kind, prefix
        self.timeout_s, self.memory_mb, self.output_chars = timeout_s, memory_mb, output_chars

    @classmethod
    def detect(cls, unsafe: bool = False, warn=print, **limits) -> "Sandbox":
        """The host's sandbox; without one, a warning and then a refusal unless unsafe is set."""
        if resource is None:
            raise SandboxUnavailable("this platform has no rlimits, so model code cannot run under limits")
        if sys.platform == "darwin" and shutil.which("sandbox-exec"):
            return cls("sandbox-exec", [shutil.which("sandbox-exec"), "-p"], **limits)
        flags = unshare_flags() if sys.platform.startswith("linux") else None
        if flags is not None:
            return cls(" ".join(["unshare", *flags[1:]]), flags, **limits)
        warn("warning: no sandbox found (sandbox-exec on macOS, unshare -n on Linux); model code would have the "
             "network and could write anywhere this user can")
        if not unsafe:
            raise SandboxUnavailable("no sandbox available; pass --unsafe-no-sandbox to run with rlimits only")
        return cls("none", [], **limits)

    def command(self, work: str) -> list[str]:
        if self.kind == "sandbox-exec":
            return [*self.prefix, mac_profile(work)]
        return list(self.prefix)

    def limits(self) -> None:
        """Runs in the child before exec: CPU seconds, file size, no core files and, where the OS allows, memory."""
        cpu = int(self.timeout_s) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        memory = self.memory_mb << 20
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        except (ValueError, OSError):
            if sys.platform != "darwin":
                raise
            # macOS refuses address-space limits; the parent's resident-memory watch holds memory there

    def run(self, code: str) -> dict:
        """Run code as main.py with only PATH in its environment; return its output, how it ended and a report."""
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="eval-run-", ignore_cleanup_errors=True) as tmp:
            work = os.path.realpath(tmp)
            with open(os.path.join(work, "main.py"), "w", encoding="utf-8") as source:
                source.write(code)
            # read back through these handles, never by path: the program could swap the files for symlinks
            with open(os.path.join(work, ".stdout"), "w+b") as out, open(os.path.join(work, ".stderr"), "w+b") as err:
                proc = subprocess.Popen([*self.command(work), sys.executable, "-I", "main.py"], cwd=work,
                                        env={"PATH": os.environ.get("PATH", os.defpath)}, stdin=subprocess.DEVNULL,
                                        stdout=out, stderr=err, preexec_fn=self.limits, start_new_session=True)
                ended = self.watch(proc, start)
                stdout, stderr = self.read(out), self.read(err)
        code = proc.returncode
        return {"stdout": stdout, "stderr": stderr, "exit": code, "ended": ended, "error": ended != "exit" or code != 0,
                "seconds": round(time.monotonic() - start, 3), "text": self.report(stdout, stderr, ended, code)}

    def watch(self, proc: subprocess.Popen, start: float) -> str:
        """Wait for the program, stopping its whole process group at the wall timeout or the memory limit."""
        ended = "exit"
        while True:
            try:
                proc.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() - start > self.timeout_s:
                ended = "timeout"
                break
            if group_rss_kib(proc.pid) > self.memory_mb << 10:
                ended = "memory"
                break
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return ended

    def read(self, handle) -> str:
        handle.seek(0)
        return handle.read(self.output_chars * 4).decode("utf-8", errors="replace")

    def report(self, stdout: str, stderr: str, ended: str, code: int) -> str:
        """The tool result the model sees: output, errors and why the program stopped, cut to the output limit."""
        parts = [stdout.rstrip()] if stdout.strip() else []
        if stderr.strip():
            parts.append("[stderr]\n" + stderr.rstrip())
        if ended == "timeout":
            parts.append(f"[stopped: over the {self.timeout_s:g} s time limit]")
        elif ended == "memory":
            parts.append(f"[stopped: over the {self.memory_mb} MiB memory limit]")
        elif code < 0:
            parts.append(f"[killed by {signal_name(code)}]")
        elif code != 0:
            parts.append(f"[exit status {code}]")
        return cut("\n".join(parts) or NO_OUTPUT, self.output_chars)
