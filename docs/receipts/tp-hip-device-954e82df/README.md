# tcp-hip at 954e82dfa

`tf-tp-test tcp-host` and `tf-tp-test tcp-hip`, two ranks on one host over 127.0.0.1 (MTU 65536), built
`zig build tf-tp-test -Doptimize=ReleaseSafe` at 954e82dfa with Zig 0.17.0. The host: AMD Ryzen AI Max+ 395, 128 GB,
Radeon 8060S (gfx1151), Linux 7.2.7, ROCm 10.0.0 (HIP 7.15). Both `tcp-hip` ranks use the one GPU, each with its own
stream and proxy. `environment.txt` holds the commit, Zig, kernel, CPU, memory, GPU, HIP runtime and loopback MTU;
`SHA256SUMS` covers the logs and `environment.txt`. This is not the two-host USB4 receipt; it shows the device side bit
for bit and its cost over `tcp-host` on one link.

| Files | Run |
|---|---|
| `tcp-host-rank*.log` | `tcp-host` on the same link, for comparison |
| `tcp-hip-rank*.log` | `tcp-hip`: every check from HIP memory, then the timings |
| `tcp-hip-killed-rank*.log` | `tcp-hip`, rank 1 killed with SIGKILL 3 s in: rank 0 ends `error: PeerFailed`, exit 1, 0.124 s after the kill |

Each rank 0 log holds the PASS line and the RESULT lines (median and p90 of 500 back-to-back calls a size; `gb_s_p50`
is one rank's shard bytes over the median). For `tcp-hip` a time is the call and its stream's synchronisation, and
`queue_p50_us` is the call alone. Each log ends with the rank's exit code; a killed rank 1 prints nothing before it
dies (exit 137).
