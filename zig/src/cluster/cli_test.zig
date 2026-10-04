//! The cluster commands end to end on fake hosts: init, check, status, serve with in-process bring-up, and node.
const std = @import("std");
const cli = @import("cli.zig");
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
    const json = try std.mem.concat(a, u8, &.{ init.out[0 .. std.mem.lastIndexOf(u8, init.out, ",\n  \"models\"").?], models });
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
    try std.testing.expectEqual(@as(u8, 3), node.code);
    try std.testing.expect(has(node.out, "s3 ") and has(node.err, "MCDMA endpoints"));
    const real = try run(a, &.{ "serve", "--cluster", file, "--model-config", glm });
    try std.testing.expectEqual(@as(u8, 3), real.code);
    try std.testing.expect(has(real.err, "nothing runs over TCP"));
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
    const dir = std.testing.environ.getPosix("TF_K3_DIR") orelse return error.SkipZigTest;
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
    try std.testing.expectEqual(@as(u8, 3), serve.code);
    try std.testing.expect(has(serve.err, "MCDMA endpoints") and has(serve.out, "node-d"));
}
