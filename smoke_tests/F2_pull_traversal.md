# F2 (security): hostile repo tree escapes the cache
pull.zig linkIntoSnapshot + blobName join the hub's raw `path` onto the
snapshot/blobs dir without normalizing `..`. A repo listing
`"path": "../../../../../pwned.txt"` writes a symlink outside HF_HUB_CACHE.
Run: ./pull_traversal.sh  (FAILs today; mock hub in malicious_hub.py)
Fix: reject tree paths containing ".." or absolute components before download/link.
