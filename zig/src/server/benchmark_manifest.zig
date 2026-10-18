//! A server-attested benchmark manifest: an attached run stays unranked unless every field passes the receipt checks.
const std = @import("std");
const json = @import("json");
const check = @import("benchmark_check.zig");
const receipt = @import("benchmark_receipt.zig");
const host = @import("benchmark_host.zig");
const stream = @import("benchmark_stream.zig");
const model_facts = @import("benchmark_model.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const Facts = struct {
    runtime: receipt.Runtime,
    hardware: receipt.Hardware,
    revision: ?[]const u8 = null,
    config_sha256: ?[]const u8 = null,
    tokenizer_sha256: ?[]const u8 = null,
    family: ?[]const u8 = null,
    quantization: ?[]const u8 = null,
    rank_count: ?u8 = null,
};

fn hexText(s: []const u8, lengths: []const usize) bool {
    for (lengths) |n| {
        if (s.len != n) continue;
        for (s) |ch| if (!(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'))) return false;
        return true;
    }
    return false;
}

fn optHex(v: Value, key: []const u8, lengths: []const usize) ?[]const u8 {
    const f = v.get(key) orelse return null;
    if (f == .null) return null;
    const s = f.str() orelse return null;
    if (!hexText(s, lengths)) return null;
    return s;
}

/// Parse a manifest body. Null means the client must keep the attached run unranked.
pub fn parse(a: Allocator, bytes: []const u8, backend: []const u8) Allocator.Error!?Facts {
    const parsed = try json.parse(a, bytes);
    const root = switch (parsed) {
        .ok => |v| v,
        .err => return null,
    };
    if (root != .object) return null;
    const schema = root.get("schema_version") orelse return null;
    if (schema.int64() != 1) return null;
    const runtime_v = root.get("runtime") orelse return null;
    if (runtime_v != .object) return null;
    const version = runtime_v.strField("tensorfold_version") orelse return null;
    const got_backend = runtime_v.strField("backend") orelse return null;
    const python = runtime_v.strField("python_version") orelse return null;
    const platform = runtime_v.strField("platform") orelse return null;
    if (!check.version(version) or !check.version(python)) return null;
    if (!std.mem.eql(u8, got_backend, backend)) return null;
    if (!std.mem.eql(u8, platform, "linux") and !std.mem.eql(u8, platform, "macos")) return null;
    const driver = if (runtime_v.get("driver")) |d| blk: {
        if (d == .null) break :blk null;
        const s = d.str() orelse return null;
        if (!check.version(s)) return null;
        break :blk s;
    } else null;
    var rank_count: ?u8 = null;
    if (root.get("rank_count")) |r| {
        if (r != .null) {
            const n = r.int64() orelse return null;
            if (n != 1 and n != 2) return null;
            rank_count = @intCast(n);
        }
    }
    const model_v = root.get("model") orelse return null;
    if (model_v != .object) return null;
    const revision = optHex(model_v, "revision", &.{ 40, 64 });
    if (model_v.get("revision")) |r| if (r != .null and revision == null) return null;
    const config_sha = optHex(model_v, "config_sha256", &.{64});
    if (model_v.get("config_sha256")) |r| if (r != .null and config_sha == null) return null;
    const tok_sha = optHex(model_v, "tokenizer_sha256", &.{64});
    if (model_v.get("tokenizer_sha256")) |r| if (r != .null and tok_sha == null) return null;
    const family = if (model_v.get("family")) |f| blk: {
        if (f == .null) break :blk null;
        const s = f.str() orelse return null;
        if (!check.name(s)) return null;
        break :blk s;
    } else null;
    const quant = if (model_v.get("quantization")) |f| blk: {
        if (f == .null) break :blk null;
        const s = f.str() orelse return null;
        if (!check.name(s)) return null;
        break :blk s;
    } else null;
    const hardware = hardwareOf(a, root.get("hardware") orelse return null) orelse return null;
    if (std.mem.eql(u8, hardware.source, "unavailable")) return null;
    return .{
        .runtime = .{
            .tensorfold_version = version,
            .backend = got_backend,
            .python_version = python,
            .platform = platform,
            .driver = driver,
        },
        .hardware = hardware,
        .revision = revision,
        .config_sha256 = config_sha,
        .tokenizer_sha256 = tok_sha,
        .family = family,
        .quantization = quant,
        .rank_count = rank_count,
    };
}

fn hardwareOf(a: Allocator, v: Value) ?receipt.Hardware {
    if (v != .object) return null;
    const source = v.strField("source") orelse return null;
    if (!std.mem.eql(u8, source, "detected") and !std.mem.eql(u8, source, "partial")) return null;
    const memory_type = v.strField("memory_type") orelse return null;
    if (!std.mem.eql(u8, memory_type, "unified") and !std.mem.eql(u8, memory_type, "dedicated") and !std.mem.eql(u8, memory_type, "unknown")) return null;
    const cpu = if (v.get("cpu_model")) |c| blk: {
        if (c == .null) break :blk null;
        const s = c.str() orelse return null;
        if (!check.name(s)) return null;
        break :blk s;
    } else null;
    const memory = if (v.get("system_memory_bytes")) |m| blk: {
        if (m == .null) break :blk null;
        const n = m.int64() orelse return null;
        if (n < 0) return null;
        break :blk @as(u64, @intCast(n));
    } else null;
    const gpus_v = v.get("gpus") orelse return null;
    if (gpus_v != .array or gpus_v.array.len > 256) return null;
    const gpus = a.alloc(receipt.Gpu, gpus_v.array.len) catch return null;
    for (gpus_v.array, gpus) |g, *slot| {
        if (g != .object) return null;
        const name = g.strField("name") orelse return null;
        if (!check.name(name)) return null;
        const mem = if (g.get("memory_bytes")) |m| blk: {
            if (m == .null) break :blk null;
            const n = m.int64() orelse return null;
            if (n < 0) return null;
            break :blk @as(?u64, @intCast(n));
        } else null;
        const cores = if (g.get("cores")) |c| blk: {
            if (c == .null) break :blk null;
            const n = c.int64() orelse return null;
            if (n < 0 or n > 100000) return null;
            break :blk @as(?u64, @intCast(n));
        } else null;
        slot.* = .{ .name = name, .memory_bytes = mem, .cores = cores };
    }
    const count = v.get("gpu_count") orelse return null;
    if (count != .null) {
        const n = count.int64() orelse return null;
        if (n != @as(i64, @intCast(gpus.len))) return null;
    }
    return .{
        .cpu_model = cpu,
        .system_memory_bytes = memory,
        .memory_type = memory_type,
        .gpus = gpus,
        .source = source,
    };
}

fn hexDigest(a: Allocator, bytes: []const u8) ![]const u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(d, .lower));
}

fn fileSha(a: Allocator, io: std.Io, path: []const u8) ?[]const u8 {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return null;
    return hexDigest(a, bytes) catch null;
}

fn rankFromEnv(env: *const std.process.Environ.Map) ?u8 {
    const raw = env.get("TF_NEMOTRON_WORLD") orelse env.get("TF_BENCHMARK_RANK_COUNT") orelse return null;
    const n = std.fmt.parseInt(u8, raw, 10) catch return null;
    if (n != 1 and n != 2) return null;
    return n;
}

fn revisionFromEnv(env: *const std.process.Environ.Map) ?[]const u8 {
    const raw = env.get("TF_CHECKPOINT_REVISION") orelse return null;
    if (!hexText(raw, &.{ 40, 64 })) return null;
    return raw;
}

/// The revision a server attests: its snapshot directory's name, as a managed run records it. ``TF_CHECKPOINT_REVISION``
/// stands in only for a directory that names none; one that disagrees with the loaded snapshot leaves the revision out.
pub fn attestedRevision(env: *const std.process.Environ.Map, model_dir: []const u8) ?[]const u8 {
    const told = revisionFromEnv(env);
    const loaded = (if (model_dir.len > 0) model_facts.snapshotRevision(model_dir) else null) orelse return told;
    if (told) |t| if (!std.mem.eql(u8, t, loaded)) return null;
    return loaded;
}

/// The receipt's name for an engine backend: the HIP engine's runs are ``rocm`` and Metal's ``mlx``, as the board names them.
pub fn receiptBackend(engine: []const u8) []const u8 {
    if (std.mem.eql(u8, engine, "hip")) return "rocm";
    if (std.mem.eql(u8, engine, "metal")) return "mlx";
    return engine;
}

/// The document a running server can attest, under the receipt's backend name. Hardware is this process's own probe.
pub fn serveDocument(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, version: []const u8, engine: []const u8, model_dir: []const u8) !Value {
    const backend = receiptBackend(engine);
    const probe = host.detect(a, io, env, backend);
    const top = try json.newObject(a);
    try top.put(a, "schema_version", try json.intValue(a, @as(u8, 1)));
    if (rankFromEnv(env)) |n| try top.put(a, "rank_count", try json.intValue(a, n));
    const runtime = try json.newObject(a);
    try runtime.put(a, "tensorfold_version", .{ .string = if (version.len > 0) version else "unknown" });
    try runtime.put(a, "backend", .{ .string = backend });
    try runtime.put(a, "python_version", .{ .string = "none" });
    try runtime.put(a, "platform", .{ .string = if (builtinTagIsMacos()) "macos" else "linux" });
    if (probe.driver) |d| try runtime.put(a, "driver", .{ .string = d });
    try top.put(a, "runtime", .{ .object = runtime });
    const model = try json.newObject(a);
    if (attestedRevision(env, model_dir)) |rev| try model.put(a, "revision", .{ .string = rev });
    if (model_dir.len > 0) {
        if (fileSha(a, io, try std.fs.path.join(a, &.{ model_dir, "config.json" }))) |h| try model.put(a, "config_sha256", .{ .string = h });
        if (model_facts.tokenizerHash(a, io, model_dir)) |h| try model.put(a, "tokenizer_sha256", .{ .string = h });
    }
    try top.put(a, "model", .{ .object = model });
    const hardware = try json.newObject(a);
    const hw = probe.hardware;
    if (hw.cpu_model) |c| try hardware.put(a, "cpu_model", .{ .string = c });
    if (hw.system_memory_bytes) |n| try hardware.put(a, "system_memory_bytes", try json.intValue(a, n));
    try hardware.put(a, "memory_type", .{ .string = hw.memory_type });
    const gpus = try a.alloc(Value, hw.gpus.len);
    for (hw.gpus, gpus) |g, *slot| {
        const one = try json.newObject(a);
        try one.put(a, "name", .{ .string = g.name });
        if (g.memory_bytes) |n| try one.put(a, "memory_bytes", try json.intValue(a, n)) else try one.put(a, "memory_bytes", .null);
        if (g.cores) |n| try one.put(a, "cores", try json.intValue(a, n)) else try one.put(a, "cores", .null);
        slot.* = .{ .object = one };
    }
    try hardware.put(a, "gpus", .{ .array = gpus });
    try hardware.put(a, "gpu_count", if (hw.gpus.len > 0) try json.intValue(a, hw.gpus.len) else .null);
    try hardware.put(a, "source", .{ .string = hw.source });
    try top.put(a, "hardware", .{ .object = hardware });
    return .{ .object = top };
}

fn builtinTagIsMacos() bool {
    return @import("builtin").os.tag == .macos;
}

test "a manifest with a GB10 and two ranks is accepted" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const body =
        \\{"schema_version":1,"rank_count":2,"runtime":{"tensorfold_version":"1.0.12","backend":"cuda","python_version":"none","platform":"linux"},"model":{"revision":"82d93699acd3be9e4617c53360fe5b3f5dbeb1b3"},"hardware":{"cpu_model":"ARM Cortex","system_memory_bytes":130000000000,"memory_type":"unified","gpus":[{"name":"NVIDIA GB10","memory_bytes":null,"cores":null}],"gpu_count":1,"source":"partial"}}
    ;
    const facts = (try parse(arena.allocator(), body, "cuda")).?;
    try std.testing.expectEqual(@as(?u8, 2), facts.rank_count);
    try std.testing.expect(std.mem.eql(u8, facts.hardware.source, "partial"));
    try std.testing.expect(facts.revision != null);
}

test "a manifest for the wrong backend is rejected" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const body =
        \\{"schema_version":1,"runtime":{"tensorfold_version":"1.0.12","backend":"mlx","python_version":"none","platform":"macos"},"model":{},"hardware":{"memory_type":"unknown","gpus":[],"gpu_count":null,"source":"partial"}}
    ;
    try std.testing.expect((try parse(arena.allocator(), body, "cuda")) == null);
}

/// Read the server's manifest. Null keeps the attached run unranked.
pub fn fetch(a: Allocator, io: std.Io, base: []const u8, backend: []const u8) !?Facts {
    const url = try std.fmt.allocPrint(a, "{s}/v1/benchmark/manifest", .{base});
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io };
    defer client.deinit();
    const got = stream.fetch(a, io, &client, .{ .url = url, .limit = 64 << 10, .timeout_s = 5 });
    if (got.failed != .none or got.status != 200 or got.body.len == 0) return null;
    return parse(a, got.body, backend);
}
