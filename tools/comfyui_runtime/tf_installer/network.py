import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .archive import parse_sums
from .pins import InstallError, MAX_ARCHIVE_BYTES, RELEASE_API, RELEASE_BASE


ALLOWED_HOSTS = frozenset({"api.github.com", "github.com", "codeload.github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"})
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def checked_url(url):
    try:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username or parts.password or parts.port not in (None, 443) or parts.fragment:
            raise InstallError("Download URL is outside the official HTTPS hosts.")
    except ValueError as error:
        raise InstallError("Malformed download URL.") from error
    return url


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        checked_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, path, limit, timeout=300, opener=None):
    checked_url(url)
    media_type = "application/vnd.github+json" if urllib.parse.urlsplit(url).hostname == "api.github.com" else "application/octet-stream"
    request = urllib.request.Request(url, headers={"User-Agent": "TensorFold-ComfyUI-installer/1", "Accept": media_type})
    opener = opener or urllib.request.build_opener(OfficialRedirects())
    deadline = time.monotonic() + timeout
    path = Path(path)
    try:
        with opener.open(request, timeout=min(30, timeout)) as response, path.open("xb") as output:
            checked_url(response.geturl())
            if response.status != 200:
                raise InstallError(f"Download returned HTTP {response.status}.")
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None and (not raw_length.isdigit() or int(raw_length) > limit):
                raise InstallError("Download Content-Length exceeds its limit or is malformed.")
            size = 0
            while True:
                if time.monotonic() > deadline:
                    raise InstallError("Download exceeded its time budget.")
                block = response.read1(min(1024 * 1024, limit - size + 1))
                if not block:
                    break
                size += len(block)
                if size > limit:
                    raise InstallError("Download exceeds its size limit.")
                output.write(block)
            if raw_length is not None and size != int(raw_length):
                raise InstallError("Download ended before its declared Content-Length.")
    except (OSError, urllib.error.URLError, ValueError) as error:
        raise InstallError(f"Download failed: {error}") from error
    return path


def version_tuple(value):
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise InstallError("Release version is not stable MAJOR.MINOR.PATCH.")
    return tuple(int(part) for part in value.split("."))


def select_release(metadata, platform):
    if not isinstance(metadata, dict) or metadata.get("draft") is not False or metadata.get("prerelease") is not False:
        raise InstallError("GitHub did not return a stable published release.")
    tag = metadata.get("tag_name", "")
    if not isinstance(tag, str) or not tag.startswith("v"):
        raise InstallError("Release tag has no v prefix.")
    version = tag[1:]
    version_tuple(version)
    filename = f"tensorfold-{version}-{platform}.tar.gz"
    assets = metadata.get("assets")
    if not isinstance(assets, list):
        raise InstallError("Release assets are missing.")
    selected = {}
    for name in (filename, filename + ".sha256"):
        matches = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == name]
        if len(matches) != 1:
            raise InstallError(f"Release needs exactly one {name} asset.")
        asset = matches[0]
        expected = f"{RELEASE_BASE}/{tag}/{name}"
        if asset.get("browser_download_url") != expected:
            raise InstallError("Release asset URL does not match its official name and tag.")
        size = asset.get("size")
        limit = MAX_ARCHIVE_BYTES if name == filename else 65536
        if type(size) is not int or not 0 < size <= limit:
            raise InstallError("Release asset size is invalid.")
        selected[name] = asset
    return {"version": version, "filename": filename, "assets": selected}


def get_release(work, platform, fetch=download):
    metadata_path = fetch(RELEASE_API, work / "release.json", 2 * 1024 * 1024)
    try:
        return select_release(json.loads(metadata_path.read_text(encoding="utf-8")), platform)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InstallError("Release metadata is not valid JSON.") from error


def fetch_release(work, release, fetch=download):
    filename = release["filename"]
    archive_asset = release["assets"][filename]
    checksum_asset = release["assets"][filename + ".sha256"]
    checksum_path = fetch(checksum_asset["browser_download_url"], work / "release.sha256", 65536)
    sums = parse_sums(checksum_path.read_text(encoding="ascii"))
    if set(sums) != {filename}:
        raise InstallError("Companion checksum must name exactly this archive.")
    digest = sums[filename]
    published = archive_asset.get("digest")
    if published is not None and published != f"sha256:{digest}":
        raise InstallError("GitHub's asset digest and companion checksum disagree.")
    archive = fetch(archive_asset["browser_download_url"], work / filename, MAX_ARCHIVE_BYTES)
    if archive.stat().st_size != archive_asset["size"]:
        raise InstallError("Downloaded archive size differs from release metadata.")
    return release["version"], archive, digest
