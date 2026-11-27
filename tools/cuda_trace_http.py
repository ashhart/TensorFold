"""Direct loopback HTTP with absolute deadlines, including headers and partial-line reads."""
from contextlib import contextmanager
import http.client
import json
import math
import socket
import threading
import time
from urllib.parse import urlsplit


def remaining(deadline):
    left = deadline - time.monotonic()
    if not math.isfinite(left) or left <= 0:
        raise TimeoutError('absolute local HTTP deadline expired')
    return left


def endpoint(base):
    url = urlsplit(base)
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port
            or url.username or url.password or url.path or url.query or url.fragment):
        raise ValueError('only the started server at http://127.0.0.1:PORT is allowed')
    return url.port


@contextmanager
def response(base, route, body=None, *, deadline=None, timeout=1800):
    port = endpoint(base)
    if route not in ('/health', '/v1/tokenize', '/v1/completions'):
        raise ValueError('unsupported local server route')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    if deadline is None:
        deadline = time.monotonic() + timeout
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=min(timeout, remaining(deadline)))
    timer = None
    reply = None
    expired = threading.Event()
    try:
        conn.connect()
        sock = conn.sock
        def expire():
            expired.set()
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        timer = threading.Timer(remaining(deadline), expire)
        timer.daemon = True
        timer.start()
        sock.settimeout(min(timeout, remaining(deadline)))
        data = None if body is None else json.dumps(body).encode()
        conn.request('GET' if body is None else 'POST', route, data, {'Content-Type': 'application/json'})
        reply = conn.getresponse()
        remaining(deadline)
        if not 200 <= reply.status < 300:
            raise ValueError(f'local HTTP status {reply.status}; redirects are refused')
        yield reply, sock, deadline
        remaining(deadline)
    except (OSError, http.client.HTTPException, ValueError) as error:
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError('absolute local HTTP deadline expired') from error
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if reply is not None:
            reply.close()
        conn.close()


def lines(reply, sock, deadline, timeout=3600):
    while True:
        sock.settimeout(min(timeout, remaining(deadline)))
        line = reply.readline()
        remaining(deadline)
        if not line:
            return
        yield line


def request(base, route, body=None, *, deadline=None, timeout=1800):
    with response(base, route, body, deadline=deadline, timeout=timeout) as (reply, sock, end):
        sock.settimeout(min(timeout, remaining(end)))
        return json.load(reply)
