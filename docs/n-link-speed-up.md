# N-link settings for speed-up mode

One logical peer can use one through eight distinct Thunderbolt RDMA devices. The MCDMA N-link build must be
installed by the owner on both ranks before using a bond. Its ABI remains 1; explicit port arrays use new optional
symbols. An older library refuses more than two devices or explicit arrays with `NeedNLinkLibrary` before verbs open.
Both bond endpoints must run the same N-aware wire build, including for a two-device bond.

Rank 0's four-device example uses a single existing meeting interface. Replace the example address and library path
with the peer's address and the owner's local build. Rank 1 uses rank 1, peer 0, and rank 0's meeting address.
Corresponding cable ends must occupy the same device-array position, even when their interface numbers differ.

```json
{"rank":0,"library":"/opt/mcdma/libmcdma-fabric.dylib","links":[
  {"peer":1,"devices":["rdma_en4","rdma_en3","rdma_en2","rdma_en13"],
   "via":"en4/192.0.2.2","ports":[7490,7491,7492,7493],"name":"speedup","gid":-1}
]}
```

`devices` or the existing `device` plus string describes physical members. `vias` supplies one or N strings, or use
`via` with one meeting interface or N plus-separated entries. One common meeting interface must expose an ACTIVE Thunderbolt RDMA device;
the QPs still use each data device's own GID. IPv4 on every data device is unnecessary if those devices expose ACTIVE
ports and usable GIDs. Run the owner's `fabric-check --gids DEVICE` probe to establish that before qualification.

`ports` and `peer_ports` are arrays of N distinct nonzero u16 UDP ports. If arrays are omitted, scalar `port` and
`peer_port` allocate N consecutive ports. Missing peer ports mirror local ports. An explicit array takes precedence
over its scalar. Default `gid` remains 1 for old one/two-device settings; it is -1 for three or more. Specify -1 to
scan each device independently, skipping zero GIDs and preferring IPv6 link-local. Invalid counts, duplicate devices,
port overflow, empty members and conflicting device/via plural and singular fields refuse parsing or endpoint creation.

The existing `tp.zig` path passes the parsed logical link through unchanged to the MCDMA endpoint. Tensor partitioning
and its flags do not depend on the number of physical members. An inactive or missing member refuses initialization;
a member failing later fails the entire logical peer. Qualify prompt first-token time and decode with token equality
before making any performance claim for this build.
