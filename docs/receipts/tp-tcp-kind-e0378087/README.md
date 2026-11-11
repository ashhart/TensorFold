# tcp-host at e0378087b

`tf-tp-test tcp-host` between two hosts, built `zig build tf-tp-test -Doptimize=ReleaseSafe` at e0378087b with Zig
0.17.0. Both hosts: AMD Ryzen AI Max+ 395, 128 GB, Linux 7.2.7. Link: USB4 networking (`thunderbolt_net`), two cables,
each its own interface and /24, MTU 1500. One TCP socket rides one cable, so the run was made once per cable, with
`TF_TP_MASTER` set to rank 0's address on that cable. No `rto_min` on the routes and no `net.ipv4.tcp_rto_min_us`, so
the kernel's 200 ms minimum RTO applied.

| Files | Run |
|---|---|
| `tcp-host-cable1-rank*.log` | cable 1 |
| `tcp-host-cable2-rank*.log` | cable 2 |
| `tcp-host-killed-rank*.log` | cable 1, rank 1 killed with SIGKILL 3 s in: rank 0 ends `error: PeerFailed`, exit 1 |
| `tcp-host-killed-timed-rank*.log` | the same, timed: each log ends with the wrapper's line; rank 0 exited 0.228 s after the kill was sent (over ssh), 3.2 s in, during the bit-for-bit pass |

Each rank 0 log holds the PASS line and the RESULT lines (median and p90 of 500 back-to-back calls a size;
`gb_s_p50` is one rank's shard bytes over the median). A killed rank 1 prints nothing before it dies.
