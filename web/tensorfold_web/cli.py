"""`tensorfold-web`: start the local server and open the page."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from . import __version__
from . import settings as settings_mod
from .engine import Engine
from .server import create_app


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["service"]:                    # `tensorfold-web service install|uninstall|status`
        from . import service

        return service.main(argv[1:])
    if argv[:1] == ["token"]:                      # `tensorfold-web token [--copy] [--new]`
        from . import access

        return access.main(argv[1:])
    p = argparse.ArgumentParser(prog="tensorfold-web", description="A local web interface for TensorFold",
                                epilog="`tensorfold-web service install` keeps it running: at login on macOS, "
                                "with systemd on Linux. `tensorfold-web token` shows the token its pages ask for.")
    p.add_argument("--host", default="127.0.0.1", help="address TensorFold Web listens on (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8090, help="port TensorFold Web listens on (default 8090)")
    p.add_argument("--engine-url", default=None, help="a TensorFold already running, here or on another machine")
    p.add_argument("--mock-engine", action="store_true", help="start a stand-in engine for development")
    p.add_argument("--mock-port", type=int, default=8080, help="the stand-in engine's port (default 8080, else a free one)")
    p.add_argument("--no-browser", action="store_true", help="do not open the page")
    p.add_argument("--token", default=None, help="required when --host is not loopback: the token the page and API calls "
                   "must carry; also worth setting when a proxy such as `tailscale serve` exposes TensorFold Web")
    p.add_argument("--token-file", default=None, help="read the token from this file instead (keeps it out of the process list)")
    p.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                   help="another name TensorFold Web may be reached under without a token (repeatable)")
    p.add_argument("--version", action="version", version=f"tensorfold-web {__version__}")
    args = p.parse_args(argv)

    if args.token_file:
        try:
            args.token = Path(args.token_file).read_text().strip() or None
        except OSError as exc:
            p.error(f"cannot read --token-file: {exc}")
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not args.token:
        p.error("binding on a non-loopback address exposes TensorFold Web's controls: pass --token <secret>, or keep "
                "127.0.0.1 and reach TensorFold Web through an SSH tunnel")
    from . import instance

    lock = instance.hold()                         # before any recovery, adoption or store: one TensorFold Web at a time
    if lock is None:
        pid = instance.holder()
        print(f"[web] another TensorFold Web is running for this user{f' (pid {pid})' if pid else ''}; it keeps the "
              "living copies, versions and settings. Stop it first (`tensorfold-web service uninstall` for the "
              "service).", file=sys.stderr, flush=True)
        # systemd does not restart on 4 (3 is uvicorn's own failed start); launchd restarts any non-zero exit, so
        # under it the exit is clean and the agent stays down until the next login, as the message says
        return 0 if os.environ.get("XPC_SERVICE_NAME", "").startswith("com.tensorfold") else 4
    s = settings_mod.load()
    eng = Engine(s, state_dir=settings_mod.config_dir())
    eng.adopt()                                    # before any override: an engine of ours keeps its address and key
    if (args.engine_url or args.mock_engine) and eng.owned:   # the same refusal as Settings
        print(f"[web] this TensorFold Web still runs {eng.model} from an earlier start; stop it (Monitor, or without "
              "--engine-url) before pointing TensorFold Web at another engine", file=sys.stderr, flush=True)
        return 1
    if args.engine_url:
        try:                                           # the same checks as Settings: no credentials in the address
            s.engine_url = settings_mod.validate({"engine_url": args.engine_url})["engine_url"]
        except ValueError as exc:
            p.error(f"--engine-url: {exc}")
        s.remote = True
    if args.mock_engine:
        from .mock_engine import serve as mock

        server = mock(port=args.mock_port)
        port = server.server_address[1]
        s.engine_url, s.remote = f"http://127.0.0.1:{port}", True
        print(f"[web] mock engine on {s.engine_url}")
    app = create_app(settings=s, engine=eng, token=args.token, bind_host=args.host, allowed_hosts=args.allow_host)
    shown = f"[{args.host}]" if ":" in args.host else args.host       # an IPv6 address needs its brackets in a URL
    url = f"http://{shown}:{args.port}/"
    # the address is printed without the token: this line ends up in service logs; only the browser opened here
    # gets the token, in its address, once
    note = " (its token is required: open it once with ?token=<the token>)" if args.token else ""
    print(f"[web] TensorFold Web {__version__} on {url}{note}", flush=True)
    if not args.no_browser:
        opened = url + (f"?token={args.token}" if args.token else "")
        threading.Timer(1.0, lambda: webbrowser.open(opened)).start()
    # an open page keeps a log stream that never ends by itself: without a bound, stopping TensorFold Web waits for it
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
