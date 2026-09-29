"""Opt-in receipt uploads with per-contributor credentials and no redirect forwarding."""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

from tensorfold.benchmark import receipt

DEFAULT_API = 'https://tensorfold.dev/api/benchmarks/v1'


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener():
    import ssl
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        pass
    else:
        # Add the packaged public CA bundle without removing OS/custom trust roots.
        context.load_verify_locations(certifi.where())
    return urllib.request.build_opener(_NoRedirect(), urllib.request.HTTPSHandler(context=context))


def api_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError('The upload API URL must not contain credentials, a query or a fragment')
    local = parsed.hostname == 'localhost'
    try:
        local = local or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        pass
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and local):
        raise ValueError('Benchmark uploads require HTTPS, except for loopback development servers')
    return value.rstrip('/')


def _token(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{16,512}', value):
        raise ValueError('Invalid benchmark upload token')
    return value


def request(base: str, route: str, token: str, *, data: dict | None = None,
            method: str | None = None) -> dict:
    base = api_url(base)
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {_token(token)}',
               'User-Agent': 'TensorFold-benchmark/1'}
    payload = None
    if data is not None:
        payload = receipt.canonical(data)
        if len(payload) > receipt.MAX_BYTES:
            raise ValueError('Upload exceeds 1 MiB')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(base+route, data=payload, headers=headers, method=method)
    try:
        with _opener().open(req, timeout=30) as response:
            if 'application/json' not in response.headers.get('Content-Type', ''):
                raise ValueError('The benchmark receiver returned a non-JSON response')
            raw = response.read(receipt.MAX_BYTES+1)
    except urllib.error.HTTPError as exc:
        messages = {401: 'Upload token was refused; run tensorfold benchmark login again',
                    403: 'This contributor cannot upload results',
                    404: 'The benchmark receiver is not available at that API URL',
                    409: 'A different receipt already uses this run identifier',
                    413: 'The benchmark receipt is too large',
                    429: 'Upload quota reached; retry later',
                    503: 'The benchmark receiver is not configured yet'}
        raise ValueError(messages.get(exc.code, f'Benchmark receiver refused the request, HTTP {exc.code}')) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ValueError('Could not reach the benchmark receiver; your local receipt is saved') from None
    if len(raw) > receipt.MAX_BYTES:
        raise ValueError('Benchmark receiver response is too large')
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError('Benchmark receiver returned invalid JSON') from None
    if not isinstance(result, dict):
        raise ValueError('Benchmark receiver returned an invalid response')
    return result


def verify_public_models(value: dict) -> None:
    """Check public metadata without using the user's Hugging Face credentials."""
    receipt.validate(value, publishing=True)
    repos = {value['model']['repo_id']}
    if value['model']['drafter_repo_id']:
        repos.add(value['model']['drafter_repo_id'])
    opener = _opener()
    for repo in repos:
        path = urllib.parse.quote(repo, safe='/')
        req = urllib.request.Request('https://huggingface.co/api/models/' + path,
                                     headers={'Accept': 'application/json', 'User-Agent': 'TensorFold-benchmark/1'})
        try:
            with opener.open(req, timeout=15) as response:
                if 'application/json' not in response.headers.get('Content-Type', ''):
                    raise ValueError('Public checkpoint metadata could not be verified')
                raw = response.read(262145)
            if len(raw) > 262144:
                raise ValueError('Public checkpoint metadata could not be verified')
            info = json.loads(raw)
        except urllib.error.HTTPError:
            raise ValueError('Only publicly accessible Hugging Face checkpoint IDs can be published') from None
        except (urllib.error.URLError, OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise ValueError('Could not verify public checkpoint metadata; the receipt remains local') from None
        if not isinstance(info, dict) or info.get('private') is not False:
            raise ValueError('Only publicly accessible Hugging Face checkpoint IDs can be published')


def auth_path() -> Path:
    return Path.home()/'.tensorfold'/'benchmarks'/'auth.json'


def login(token: str, base: str = DEFAULT_API) -> dict:
    token = _token(token.strip())
    base = api_url(base)
    profile = request(base, '/me', token)
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, name = tempfile.mkstemp(prefix='.auth-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump({'api': base, 'token': token}, handle)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return profile


def credential(base: str = DEFAULT_API) -> str:
    override = os.environ.get('TENSORFOLD_BENCHMARK_TOKEN')
    if override:
        return _token(override)
    try:
        path = auth_path()
        if path.stat().st_mode & 0o077:
            raise ValueError('Upload credentials have broad file permissions; run benchmark login again')
        if path.stat().st_size > 4096:
            raise ValueError('Invalid benchmark credentials')
        stored = json.loads(path.read_text())
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError('An upload token is required; run tensorfold benchmark login') from None
    if not isinstance(stored, dict) or set(stored) != {'api', 'token'}:
        raise ValueError('Invalid benchmark credentials')
    if stored.get('api') != api_url(base):
        raise ValueError('Upload credentials belong to a different API URL; log in to this receiver first')
    return _token(stored.get('token'))


def upload(value: dict, base: str = DEFAULT_API) -> dict:
    verify_public_models(value)
    result = request(base, '/submissions', credential(base), data=value)
    if not isinstance(result.get('id'), str) or result.get('status') not in ('pending', 'approved'):
        raise ValueError('Benchmark receiver did not acknowledge a valid submission')
    return result
