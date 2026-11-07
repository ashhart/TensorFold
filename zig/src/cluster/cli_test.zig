//! The cluster commands end to end on fake hosts: init, check, status, serve with in-process bring-up, and node.
const std = @import("std");
const platform = @import("sys");
const cli = @import("cli.zig");
const config = @import("config.zig");
const mcdma_link = @import("mcdma_link.zig");
const fixtures = @import("probe_fixtures.zig");

const io = std.testing.io;

/// GLM-5.3's config as the lead gave it; head dims, vocab and indexer heads assumed as DeepSeek-V3.2's.
pub const glm53_config =
    \\{ "model_type": "glm_moe_dsa", "hidden_size": 6144, "num_hidden_layers": 78, "vocab_size": 151552,
    \\  "num_attention_heads": 64, "num_key_value_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512,
    \\  "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256, "n_routed_experts": 256,
    \\  "num_experts_per_tok": 8, "n_shared_experts": 1, "moe_intermediate_size": 2048, "first_k_dense_replace": 3,
    \\  "intermediate_size": 12288, "num_nextn_predict_layers": 1, "index_topk": 2048, "index_n_heads": 32,
    \\  "index_head_dim": 128, "max_position_embeddings": 202752,
    \\  "quantization_config": { "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128] } }
;

const Run = struct { out: []const u8, err: []const u8, code: u8 };

fn run(a: std.mem.Allocator, args: []const []const u8) !Run {
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    var sys: cli.System = .{ .io = io };
    const code = try cli.run(.{ .io = io, .a = a, .out = &out.writer, .err = &err.writer, .runner = sys.runner() }, args);
    return .{ .out = out.written(), .err = err.written(), .code = code };
}

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}

test "only cluster, node and serve --cluster belong to the cluster commands" {
    try std.testing.expect(cli.wants(&.{ "cluster", "check" }) and cli.wants(&.{ "node", "--name", "x" }));
    try std.testing.expect(cli.wants(&.{ "serve", "m", "--cluster", "c.json" }) and cli.wants(&.{ "serve", "--cluster=c.json" }));
    try std.testing.expect(!cli.wants(&.{ "serve", "m", "--port", "9" }) and !cli.wants(&.{ "run", "m" }) and !cli.wants(&.{}));
}

test "four fake hosts: init discovers the mesh, check plans GLM-5.3, serve brings the cluster up in-process" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const names = [_][]const u8{ "s1", "s2", "s3", "s4" };
    try fixtures.writeHosts(io, a, tmp.dir, &names, &fixtures.mesh4);
    const init = try run(a, &.{ "cluster", "init", "--node", "s1=192.0.2.10", "--node", "s2=s2.local@s1", "--node", "s3=s3.local@s1", "--node", "s4=s4.local@s1", "--ssh-user", "tf", "--ssh-key", "/keys/k", "--name", "lab", "--probes", root });
    try std.testing.expectEqual(@as(u8, 0), init.code);
    try std.testing.expect(has(init.out, "\"peer\": \"s4\""));
    const models = ", \"models\": { \"glm-5.3\": { \"path\": \"/models/glm53\", \"parallel\": { \"tensor\": 4, \"expert\": 4 }, \"streams\": 32, \"context\": 8192 } } }\n";
    const json = try std.mem.concat(a, u8, &.{ init.out[0..std.mem.lastIndexOf(u8, init.out, ",\n  \"models\"").?], models });
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = json });
    try tmp.dir.writeFile(io, .{ .sub_path = "glm.json", .data = glm53_config });
    const file = try std.fs.path.join(a, &.{ root, "cluster.json" });
    const glm = try std.fs.path.join(a, &.{ root, "glm.json" });
    const check = try run(a, &.{ "cluster", "check", "--cluster", file, "--model-config", glm, "--probes", root });
    try std.testing.expectEqual(@as(u8, 0), check.code);
    try std.testing.expect(has(check.out, "layout: tensor 4 x pipeline 1") and has(check.out, "admission: experts spread, MLA by") and has(check.out, "128 rows"));
    const status = try run(a, &.{ "cluster", "status", "--cluster", file, "--model-config", glm, "--probes", root });
    try std.testing.expect(has(status.out, "shape: mesh, 6 links"));
    const serve = try run(a, &.{ "serve", "--cluster", file, "--model-config", glm, "--probes", root });
    try std.testing.expectEqual(@as(u8, 0), serve.code);
    try std.testing.expect(has(serve.out, "fake cluster up: 4 nodes agree on plan"));
    const launches = try tmp.dir.readFileAlloc(io, "launches.log", a, .limited(1 << 20));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, launches, "tensorfold node --name"));
    try std.testing.expect(has(launches, "ProxyCommand=ssh -i /keys/k -o IdentitiesOnly=yes -o BatchMode=yes -W %h:%p tf@192.0.2.10"));
    const node = try run(a, &.{ "node", "--name", "s3", "--cluster", file, "--probes", root });
    try std.testing.expectEqual(@as(u8, 1), node.code);
    try std.testing.expect(has(node.out, "s3 ") and has(node.err, "fake hosts (--probes) have no MCDMA links"));
    // converged on real hosts: refused before any node starts, naming the missing Shard
    const real = try run(a, &.{ "serve", "--cluster", file, "--model-config", glm });
    try std.testing.expectEqual(@as(u8, 1), real.code);
    try std.testing.expect(has(real.err, "real backend.Shard") and has(real.err, "no node was started"));
    try std.testing.expect(std.mem.indexOf(u8, real.out, "starts tensorfold node") == null);
    try std.testing.expectEqual(@as(u8, 0), (try run(a, &.{ "serve", "--cluster", file, "--model-config", glm, "--dry-run" })).code);
    const real_node = try run(a, &.{ "node", "--name", "s3", "--cluster", file });
    try std.testing.expectEqual(@as(u8, 1), real_node.code);
    try std.testing.expect(has(real_node.err, "real backend.Shard"));
    try std.testing.expectEqual(@as(u8, 2), (try run(a, &.{"bogus"})).code);
}

const studios =
    \\{ "schema": "tensorfold-cluster/1", "cluster": "studios", "ssh": { "user": "tf", "key": "/keys/cluster" },
    \\  "nodes": [ { "name": "node-a", "address": "192.0.2.10", "backend": "metal", "chip": "Apple M3 Ultra", "memory_gb": 512 },
    \\             { "name": "node-b", "address": "node-b", "via": "node-a", "backend": "metal", "chip": "Apple M3 Ultra", "memory_gb": 512 },
    \\             { "name": "node-c", "address": "node-c", "via": "node-a", "backend": "metal", "chip": "Apple M3 Ultra", "memory_gb": 512 },
    \\             { "name": "node-d", "address": "node-d", "via": "node-a", "backend": "metal", "chip": "Apple M3 Ultra", "memory_gb": 512, "gpu_limit_gb": 300 } ],
    \\  "models": { "kimi-k3": { "path": "/models/Kimi-K3", "parallel": { "tensor": 4, "pipeline": 1, "expert": 4 }, "drafter": { "path": "/models/k3-drafter", "size_gb": 4 } } } }
;

test "check plans K3 from its config and Hub file list and names the raise a low node needs; serve stops before launching" {
    const dir = platform.getenv("TF_K3_DIR") orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = studios });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/cluster.json", .{tmp.sub_path});
    const cfg = try std.fs.path.join(a, &.{ dir, "config.json" });
    const files = try std.fs.path.join(a, &.{ dir, "files.tsv" });
    const check = try run(a, &.{ "cluster", "check", "--cluster", path, "--model-config", cfg, "--model-files", files });
    try std.testing.expectEqual(@as(u8, 1), check.code);
    try std.testing.expect(has(check.out, "sudo sysctl iogpu.wired_limit_mb=") and has(check.out, "node node-d"));
    try std.testing.expectEqual(@as(u8, 1), (try run(a, &.{ "serve", "kimi-k3", "--cluster", path, "--model-config", cfg, "--model-files", files })).code);
    const fixed = try std.mem.replaceOwned(u8, a, studios, ", \"gpu_limit_gb\": 300", "");
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = fixed });
    const serve = try run(a, &.{ "serve", "--cluster", path, "--model-config", cfg, "--model-files", files });
    try std.testing.expectEqual(@as(u8, 1), serve.code);
    try std.testing.expect(has(serve.err, "real backend.Shard") and has(serve.out, "node-d"));
}

test "a disaggregated model on fake hosts: no tensor-parallel group across its two lists, no model files, the plan the nodes derive" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    // s1 is a Mac (macOS probes); s2 a GB10 on Linux tools: meminfo, nvidia-smi, a ConnectX-7's ibv_devinfo -v
    try fixtures.writeHosts(io, a, tmp.dir, &.{"s1"}, &fixtures.mesh4);
    try tmp.dir.createDirPath(io, "s2");
    const gb10 = [_][2][]const u8{
        .{ "s2/meminfo.txt", "MemTotal:       131072000 kB\nMemAvailable:    80000000 kB\n" },
        .{ "s2/nvidia.txt", "NVIDIA GB10, [N/A], [N/A]\n" },
        .{ "s2/ibv_devinfo.txt", @embedFile("testdata/gb10-ibv.txt") },
        .{ "s2/df-root.txt", "Filesystem     1K-blocks      Used Available Use% Mounted on\n/dev/nvme0n1p2 900000000 100000000 750000000  12% /\n" },
        .{ "s2/machine-id.txt", "0123456789abcdef0123456789abcdef\n" },
    };
    for (gb10) |f| try tmp.dir.writeFile(io, .{ .sub_path = f[0], .data = f[1] });
    // the model's path is nowhere: a disaggregated bring-up plans no tensor placement, so it reads no model files
    const head =
        \\{ "schema": "tensorfold-cluster/1", "cluster": "pair",
        \\  "nodes": [ { "name": "s1", "address": "192.0.2.10", "backend": "metal" },
        \\             { "name": "s2", "address": "192.0.2.11", "backend": "cuda", "gpu_limit_gb": 100 } ],
        \\  "models": { "q": { "path": "/nonexistent/q", "mode": "disaggregated", "prefill": ["s2"], "decode": ["s1"] } }
    ;
    const linked = head ++
        \\,
        \\  "links": [ { "name": "s1-s2", "kind": "cx5", "mtu": 1024,
        \\    "ends": [ { "node": "s1", "device": "rdma_mcrdma0", "via": "en7/192.0.2.11", "port": 7490, "gid": 0 },
        \\              { "node": "s2", "device": "rocep1s0f0", "via": "enp1s0f0np0/192.0.2.10", "port": 7491, "gid": 1 } ] } ] }
    ;
    const file = try std.fs.path.join(a, &.{ root, "cluster.json" });
    // a GB10 prints no memory: without gpu_limit_gb it is refused, exit 1 and no stack trace
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = try std.mem.replaceOwned(u8, a, linked, ", \"gpu_limit_gb\": 100", "") });
    try std.testing.expectEqual(@as(u8, 1), (try run(a, &.{ "serve", "--cluster", file, "--probes", root })).code);
    // no link declared: the probed topology has no Mac to GB10 cable (a GB10 reports no Thunderbolt peers)
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = head ++ " }\n" });
    const unlinked = try run(a, &.{ "serve", "--cluster", file, "--probes", root });
    try std.testing.expectEqual(@as(u8, 1), unlinked.code);
    try std.testing.expect(has(unlinked.err, "the fake cluster did not come up: NeverSettled"));
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = linked });
    tmp.dir.deleteFile(io, "launches.log") catch {};
    const serve = try run(a, &.{ "serve", "--cluster", file, "--probes", root });
    if (serve.code != 0) std.debug.print("{s}{s}", .{ serve.out, serve.err });
    try std.testing.expectEqual(@as(u8, 0), serve.code);
    var probs: config.Problems = .{ .a = a };
    const c = try config.parse(a, linked, &probs);
    const m = c.model("q").?;
    const want = try std.fmt.allocPrint(a, "fake cluster up: 2 nodes agree on disaggregated plan {x:0>16} (prefill on 1, decode on 1;", .{mcdma_link.planDigest(c, m, try mcdma_link.ranks(a, m), .{})});
    try std.testing.expect(has(serve.out, want));
    try std.testing.expect(std.mem.indexOf(u8, serve.out, "layout: tensor") == null);
    // the leader (s1, the model's first node) starts the one other rank
    const launches = try tmp.dir.readFileAlloc(io, "launches.log", a, .limited(1 << 20));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, launches, "tensorfold node --name"));
    try std.testing.expect(has(launches, "--name s2 --leader s1 --model q"));
}

/// A Mac decoding for a GB10 on real hosts, with `links` and the Mac's `library` as given.
fn pairFile(a: std.mem.Allocator, links: []const u8, library: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a,
        \\{{ "schema": "tensorfold-cluster/1", "cluster": "pair",
        \\  "nodes": [ {{ "name": "mac", "address": "192.0.2.1", "backend": "metal", "memory_gb": 128{s} }},
        \\             {{ "name": "gb10", "address": "192.0.2.2", "backend": "cuda", "memory_gb": 128, "library": "/opt/mcdma/libmcdma-fabric.so" }},
        \\             {{ "name": "spare", "address": "192.0.2.3", "backend": "metal", "memory_gb": 128 }} ]{s},
        \\  "models": {{ "q": {{ "path": "/models/q", "nodes": ["mac", "gb10"], "mode": "disaggregated", "prefill": ["gb10"], "decode": ["mac"] }},
        \\              "k": {{ "path": "/models/k", "nodes": ["mac", "spare"] }} }} }}
    , .{ library, links });
}

const cx_link =
    \\,
    \\  "links": [ { "name": "mb", "kind": "cx5", "mtu": 1024,
    \\    "ends": [ { "node": "mac", "device": "rdma_mcrdma0", "via": "en7/192.0.2.2", "port": 7490, "gid": 0 },
    \\              { "node": "gb10", "device": "rocep1s0f1", "via": "enp1/192.0.2.1", "port": 7491, "gid": 1 } ] } ]
;

test "real hosts: a disaggregated model refuses by name before any node starts, exit 3 for MCDMA's library, and a dry run shows each link" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/cluster.json", .{tmp.sub_path});
    const Case = struct { links: []const u8, library: []const u8, args: []const []const u8, code: u8, err: []const u8 };
    const serve: []const []const u8 = &.{ "serve", "q", "--cluster", file, "--leader", "mac" };
    const node: []const []const u8 = &.{ "node", "--name", "mac", "--model", "q", "--cluster", file };
    const cases = [_]Case{
        .{ .links = "", .library = ", \"library\": \"/bin/sh\"", .args = serve, .code = 1, .err = "no link between gb10 and mac" },
        .{ .links = cx_link, .library = "", .args = serve, .code = 3, .err = "mac names no MCDMA library" },
        .{ .links = cx_link, .library = ", \"library\": \"/nonexistent/libmcdma-fabric.dylib\"", .args = serve, .code = 3, .err = "no MCDMA library at /nonexistent/libmcdma-fabric.dylib" },
        .{ .links = cx_link, .library = "", .args = node, .code = 3, .err = "mac names no MCDMA library" },
        .{ .links = cx_link, .library = ", \"library\": \"/nonexistent/libmcdma-fabric.dylib\"", .args = node, .code = 3, .err = "no MCDMA library at" },
        .{ .links = cx_link, .library = "", .args = &.{ "serve", "q", "--cluster", file, "--leader", "spare" }, .code = 1, .err = "--leader spare is not one of model q's prefill or decode nodes" },
        .{ .links = cx_link, .library = "", .args = &.{ "node", "--name", "spare", "--model", "q", "--cluster", file }, .code = 1, .err = "spare is not one of model q's prefill or decode nodes" },
        .{ .links = cx_link, .library = "", .args = &.{ "node", "--name", "mac", "--model", "k", "--cluster", file }, .code = 1, .err = "real backend.Shard" },
        .{ .links = cx_link, .library = "", .args = &.{ "serve", "k", "--cluster", file }, .code = 1, .err = "real backend.Shard" },
    };
    for (cases) |k| {
        try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = try pairFile(a, k.links, k.library) });
        const got = try run(a, k.args);
        if (got.code != k.code) std.debug.print("{s}: exit {d}\n{s}{s}", .{ k.err, got.code, got.out, got.err });
        try std.testing.expectEqual(k.code, got.code);
        try std.testing.expect(has(got.err, k.err));
        try std.testing.expect(std.mem.indexOf(u8, got.out, "starts tensorfold node") == null);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = try pairFile(a, cx_link, ", \"library\": \"/bin/sh\"") });
    try std.testing.expectEqual(@as(u8, 2), (try run(a, &.{ "serve", "q", "--cluster", file, "--leader", "mac", "--hold", "soon" })).code);
    const dry = try run(a, &.{ "serve", "q", "--cluster", file, "--leader", "mac", "--dry-run" });
    try std.testing.expectEqual(@as(u8, 0), dry.code);
    try std.testing.expect(has(dry.out, "starts tensorfold node over key-only ssh on: gb10\n"));
    try std.testing.expect(has(dry.out, "mac: rank 1 of 2 (0 gb10 cuda, 1 mac metal); plan "));
    try std.testing.expect(has(dry.out, "link \"mb\" to gb10: rdma_mcrdma0 via en7/192.0.2.2, port 7490 to 7491, gid 0, path MTU 1024, kind cx5"));
    try std.testing.expect(has(dry.out, "dry run: no node was started and no link opened"));
}

test "check, status and serve fail when a model cannot be planned" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = studios });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.json", .data = "{\"model_type\":\"unsupported\",\"hidden_size\":128,\"num_hidden_layers\":1}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "files.tsv", .data = "model.safetensors\t1024\n" });
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const file = try std.fs.path.join(a, &.{ root, "cluster.json" });
    const shape = try std.fs.path.join(a, &.{ root, "model.json" });
    const files = try std.fs.path.join(a, &.{ root, "files.tsv" });
    const serve = try run(a, &.{ "serve", "kimi-k3", "--cluster", file, "--model-config", shape, "--dry-run" });
    try std.testing.expect(has(serve.out, "not planned: FileListNeeded"));
    try std.testing.expectEqual(@as(u8, 1), serve.code);
    for ([_][]const u8{ "check", "status" }) |command| {
        const missing = try run(a, &.{ "cluster", command, "--cluster", file, "--model-config", shape });
        try std.testing.expect(has(missing.out, "not planned: FileListNeeded"));
        try std.testing.expectEqual(@as(u8, 1), missing.code);
        const unsupported = try run(a, &.{ "cluster", command, "--cluster", file, "--model-config", shape, "--model-files", files });
        try std.testing.expect(has(unsupported.out, "not planned: EstimateUnsupported"));
        try std.testing.expectEqual(@as(u8, 1), unsupported.code);
    }
}

test "serve MODEL checks only the model it serves: another model in the file that cannot be planned does not stop it" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/cluster.json", .{tmp.sub_path});
    // model k (converged, mac and spare) cannot be planned: spare declares no memory
    const text = try std.mem.replaceOwned(u8, a, try pairFile(a, cx_link, ", \"library\": \"/bin/sh\""), "\"192.0.2.3\", \"backend\": \"metal\", \"memory_gb\": 128", "\"192.0.2.3\", \"backend\": \"metal\"");
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = text });
    try std.testing.expectEqual(@as(u8, 1), (try run(a, &.{ "cluster", "check", "--cluster", file })).code);
    const got = try run(a, &.{ "serve", "q", "--cluster", file, "--leader", "mac", "--dry-run" });
    if (got.code != 0) std.debug.print("exit {d}\n{s}{s}", .{ got.code, got.out, got.err });
    try std.testing.expectEqual(@as(u8, 0), got.code);
    try std.testing.expect(has(got.out, "model q: disaggregated"));
    try std.testing.expect(std.mem.indexOf(u8, got.out, "model k:") == null);
}

test "serve refuses before starting any node when another node's library, link or window is wrong, not only its own" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/cluster.json", .{tmp.sub_path});
    const serve: []const []const u8 = &.{ "serve", "q", "--cluster", file, "--leader", "mac", "--dry-run" };
    // the GB10 names no library: its node would stop with exit 3 after serve had started it
    const bare = try std.mem.replaceOwned(u8, a, try pairFile(a, cx_link, ", \"library\": \"/bin/sh\""), ", \"library\": \"/opt/mcdma/libmcdma-fabric.so\"", "");
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = bare });
    const lib = try run(a, serve);
    try std.testing.expectEqual(@as(u8, 3), lib.code);
    try std.testing.expect(has(lib.err, "gb10 names no MCDMA library"));
    try std.testing.expect(std.mem.indexOf(u8, lib.out, "starts tensorfold node") == null);
    // two GB10s prefill for the Mac: the Mac reaches both, but nothing joins the two GB10s
    const three =
        \\{ "schema": "tensorfold-cluster/1", "cluster": "three",
        \\  "nodes": [ { "name": "mac", "address": "192.0.2.1", "backend": "metal", "library": "/bin/sh" },
        \\             { "name": "gb10", "address": "192.0.2.2", "backend": "cuda", "library": "/opt/mcdma/libmcdma-fabric.so" },
        \\             { "name": "gb10b", "address": "192.0.2.4", "backend": "cuda", "library": "/opt/mcdma/libmcdma-fabric.so" } ],
        \\  "links": [ { "name": "mb", "kind": "cx5", "ends": [ { "node": "mac", "device": "rdma_mcrdma0", "via": "en7/192.0.2.2", "port": 7490, "gid": 0 }, { "node": "gb10", "device": "r0", "via": "e0/192.0.2.1", "port": 7491, "gid": 1 } ] },
        \\             { "name": "mb2", "kind": "cx5", "ends": [ { "node": "mac", "device": "rdma_mcrdma1", "via": "en8/192.0.2.4", "port": 7492, "gid": 0 }, { "node": "gb10b", "device": "r0", "via": "e0/192.0.2.1", "port": 7493, "gid": 1 } ] } ],
        \\  "models": { "q": { "path": "/models/q", "mode": "disaggregated", "prefill": ["gb10", "gb10b"], "decode": ["mac"] } } }
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = three });
    const link = try run(a, serve);
    try std.testing.expectEqual(@as(u8, 1), link.code);
    try std.testing.expect(has(link.err, "no link between gb10 and gb10b"));
    try std.testing.expect(std.mem.indexOf(u8, link.out, "starts tensorfold node") == null);
}

test "node and serve refuse a flag they do not take by name, before reading probes or opening anything" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/cluster.json", .{tmp.sub_path});
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = try pairFile(a, cx_link, "") });
    const Case = struct { args: []const []const u8, err: []const u8 };
    const cases = [_]Case{
        // node has no dry run: it used to open the real endpoint and wait for its peer
        .{ .args = &.{ "node", "--name", "mac", "--model", "q", "--cluster", file, "--dry-run" }, .err = "node: unknown flag --dry-run" },
        .{ .args = &.{ "node", "--name", "mac", "--cluster", file, "--hold", "5" }, .err = "node: unknown flag --hold" },
        .{ .args = &.{ "node", "mac", "--cluster", file }, .err = "node: unexpected argument mac" },
        .{ .args = &.{ "node", "--cluster", file, "--name" }, .err = "node: --name needs a value" },
        .{ .args = &.{ "serve", "q", "--cluster", file, "--leader", "mac", "--dryrun" }, .err = "serve: unknown flag --dryrun" },
        .{ .args = &.{ "serve", "q", "--cluster", file, "--dry-run=yes" }, .err = "serve: --dry-run takes no value" },
        .{ .args = &.{ "serve", "q", "k", "--cluster", file }, .err = "serve: unexpected argument k" },
    };
    for (cases) |k| {
        const got = try run(a, k.args);
        if (got.code != 2) std.debug.print("{s}: exit {d}\n{s}{s}", .{ k.err, got.code, got.out, got.err });
        try std.testing.expectEqual(@as(u8, 2), got.code);
        try std.testing.expect(has(got.err, k.err));
        try std.testing.expectEqualStrings("", got.out);
    }
    // both spellings of a flag's value still pass; the file's missing library is the next refusal
    try std.testing.expectEqual(@as(u8, 3), (try run(a, &.{ "node", "--name=mac", "--model", "q", try std.fmt.allocPrint(a, "--cluster={s}", .{file}), "--leader", "gb10" })).code);
}
