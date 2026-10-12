//! Checkpoint facts for a receipt: repo, revision, family, file hashes and the quantization label.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const check = @import("benchmark_check.zig");
const receipt = @import("benchmark_receipt.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const Model = receipt.Model;

const config_limit = 4 << 20;
const tokenizer_limit = 256 << 20;
const tokenizer_files = [_][]const u8{ "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja" };

/// ``config.json``'s bytes and its parsed value.
pub const Config = struct { bytes: []const u8, value: Value };

/// Null when missing or past 4 MiB; ``value`` is ``.null`` when the file is not a JSON object.
pub fn readConfig(a: Allocator, io: std.Io, dir: []const u8) ?Config {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return null;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(config_limit + 1)) catch return null;
    const parsed = json.parse(a, bytes) catch return null;
    return switch (parsed) {
        .ok => |v| .{ .bytes = bytes, .value = if (v == .object) v else .null },
        .err => .{ .bytes = bytes, .value = .null },
    };
}

/// Checkpoint facts; a local directory gives hashes and labels but no name; ``drafter`` is a repo id or null.
pub fn describe(a: Allocator, io: std.Io, target: []const u8, dir: []const u8, local_dir: bool, families: []const api.Family, drafter: ?[]const u8) !Model {
    const repo: ?[]const u8 = if (!local_dir and check.repoId(target)) target else null;
    var out: Model = .{ .repo_id = repo, .local = repo == null };
    const name = std.fs.path.basename(dir);
    const parent = std.fs.path.basename(std.fs.path.dirname(dir) orelse "");
    if (repo != null and std.mem.eql(u8, parent, "snapshots") and (name.len == 40 or name.len == 64) and lowerHex(name)) out.revision = name;
    const config = readConfig(a, io, dir);
    if (config) |c| out.config_sha256 = try hexDigest(a, c.bytes);
    const value: Value = if (config) |c| c.value else .null;
    if (modelType(value)) |kind| for (families) |f| {
        if (std.mem.eql(u8, f.model_type, kind)) out.family = kind;
    };
    out.quantization = try quantization(a, value);
    out.tokenizer_sha256 = tokenizerHash(a, io, dir);
    if (drafter) |d| if (check.repoId(d)) {
        out.drafter_repo_id = d;
    };
    return out;
}

/// The bytes of the ``*.safetensors`` files in ``dir``, through a cache snapshot's links; 0 when unreadable.
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var it = d.iterate();
    var total: u64 = 0;
    while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const st = d.statFile(io, e.name, .{}) catch continue;
        if (st.kind == .file) total +|= st.size;
    }
    return total;
}

/// An attached server's facts: only the public ID it was asked for.
pub fn attached(target: []const u8, target_is_path: bool) Model {
    const public = !target_is_path and check.repoId(target);
    return .{ .repo_id = if (public) target else null, .local = !public };
}

fn lowerHex(s: []const u8) bool {
    for (s) |ch| if (!(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'))) return false;
    return true;
}

fn hexDigest(a: Allocator, bytes: []const u8) ![]const u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(d, .lower));
}

/// ``model_type``, else ``text_config.model_type``, when it is a non-empty string.
pub fn modelType(config: Value) ?[]const u8 {
    if (config.get("model_type")) |t| if (t.truthy()) return t.str();
    const text = config.get("text_config") orelse return null;
    const t = text.get("model_type") orelse return null;
    return t.str();
}

/// A drafter made for one family and shape: the target's hidden size and layer count.
const Known = struct { model_type: []const u8, hidden: i64, layers: i64, repo: []const u8 };
const known_drafters = [_]Known{.{ .model_type = "qwen3_5", .hidden = 5120, .layers = 64, .repo = "z-lab/Qwen3.8-27B-DFlash2" }};

/// The repo ID of the drafter made for this checkpoint's family and shape (Qwen3.8-27B's DFlash2), else null.
pub fn knownDrafter(config: Value) ?[]const u8 {
    const kind = modelType(config) orelse return null;
    for (known_drafters) |d| {
        if (!std.mem.eql(u8, kind, d.model_type)) continue;
        for ([_]?Value{ config, config.get("text_config") }) |source| {
            const src = source orelse continue;
            const hidden = (src.get("hidden_size") orelse continue).int64() orelse continue;
            const layers = (src.get("num_hidden_layers") orelse continue).int64() orelse continue;
            if (hidden == d.hidden and layers == d.layers) return d.repo;
        }
    }
    return null;
}

const methods = [_][]const u8{ "mlx", "exl3", "modelopt", "compressed-tensors", "awq", "gptq", "bitsandbytes" };
const details = [_][]const u8{ "nvfp4", "mxfp4", "mxfp8", "fp8", "fp4", "mixed_precision", "affine", "float-quantized", "pack-quantized", "int-quantized", "w4a16_nvfp4" };

/// The quantization label (``mlx-4bit-g64``): an allowlisted method, details, bits and group; else null.
pub fn quantization(a: Allocator, config: Value) !?[]const u8 {
    if (config != .object) return null;
    const text = config.get("text_config");
    for ([_]?Value{ config, if (text != null and text.? == .object) text.? else null }) |source| {
        const src = source orelse continue;
        for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
            const q = src.get(key) orelse continue;
            if (q != .object or q.object.count() == 0) continue;
            const method = if (q.get("quant_method")) |m| (if (m == .string) try std.ascii.allocLowerString(a, m.string) else "") else if (q.has("bits")) "mlx" else "";
            if (!oneOf(method, &methods)) return null;
            var parts: std.ArrayList([]const u8) = .empty;
            try parts.append(a, method);
            for ([_][]const u8{ "quant_algo", "format", "mode" }) |k| if (q.get(k)) |v| if (v == .string) {
                const lower = try std.ascii.allocLowerString(a, v.string);
                if (oneOf(lower, &details)) try parts.append(a, lower);
            };
            if (q.get("bits")) |b| if (b.int64()) |n| if (n >= 1 and n <= 32) try parts.append(a, try std.fmt.allocPrint(a, "{d}bit", .{n}));
            if (q.get("group_size")) |g| if (g.int64()) |n| if (n >= 1 and n <= 4096) try parts.append(a, try std.fmt.allocPrint(a, "g{d}", .{n}));
            return try std.mem.join(a, "-", parts.items);
        }
    }
    return if (config.object.count() > 0) "unquantized" else null;
}

fn oneOf(s: []const u8, options: []const []const u8) bool {
    for (options) |o| if (std.mem.eql(u8, s, o)) return true;
    return false;
}

/// sha256 over the tokenizer files present, each framed ``name NUL size NUL``; null when none or past 256 MiB.
pub fn tokenizerHash(a: Allocator, io: std.Io, dir: []const u8) ?[]const u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var total: u64 = 0;
    var found: usize = 0;
    const chunk = a.alloc(u8, 1 << 20) catch return null;
    for (tokenizer_files) |name| {
        const path = std.fs.path.join(a, &.{ dir, name }) catch return null;
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch continue;
        defer file.close(io);
        const st = file.stat(io) catch return null;
        if (st.kind != .file) continue;
        total += st.size;
        if (total > tokenizer_limit) return null;
        var head: [64]u8 = undefined;
        h.update(std.fmt.bufPrint(&head, "{s}\x00{d}\x00", .{ name, st.size }) catch return null);
        var seen: u64 = 0;
        while (true) {
            const n = file.readPositional(io, &.{chunk}, seen) catch return null;
            if (n == 0) break;
            seen += n;
            if (seen > st.size) return null;
            h.update(chunk[0..n]);
        }
        if (seen != st.size) return null;
        found += 1;
    }
    if (found == 0) return null;
    return a.dupe(u8, &std.fmt.bytesToHex(h.finalResult(), .lower)) catch null;
}
