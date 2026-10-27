//! The control vector a serve starts with: the file it reads, the checks that refuse one, and the line that reports it.

const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const glp = api.glp;
const log = @import("log.zig");
const cfg = @import("checkpoint_config");
const cli_flag = @import("cli_flag.zig");

/// The flags the control vector adds to the server's own.
pub const flags = [_]cli_flag.Flag{
    .{ .name = "--glp", .native = true },
    .{ .name = "--glp-alpha", .native = true },
};

/// The model type of the family that takes a control vector: Qwen3.8-27B.
const steer_type = "qwen3_5";

/// The hidden size of that family: Qwen3.8-27B is 5120 wide.
const steer_hidden: usize = 5120;

/// The bytes a control vector file may hold; a 64-layer 5120-wide vector is about 1.3 MiB.
const file_limit: usize = 64 << 20;

/// The key a file that claims its own content digest carries.
const digest_key = "glp.content_sha256";

/// What the --glp flags hold: the file the operator named, the strength and the vector the file carries.
pub const Vec = struct {
    /// The file's path, as the operator wrote it (null: no file named).
    path: ?[]const u8 = null,
    /// The strength the operator gave (null: the file's own alpha_default is in force).
    alpha: ?f64 = null,
    /// The vector the file carries, read once at startup (null: the file was not read).
    vector: ?glp.Vector = null,
    /// Whether the file claims a content digest the reader checks.
    claimed: bool = false,
};

/// The --glp file with the path the operator gave; null when the operator named none.
pub fn withPath(g: ?Vec, path: []const u8) Vec {
    return .{ .path = path, .alpha = if (g) |v| v.alpha else null };
}

/// The --glp file with the strength the operator gave; null when the operator named none.
pub fn withAlpha(g: ?Vec, alpha: f64) Vec {
    return .{ .path = if (g) |v| v.path else null, .alpha = alpha };
}

/// A finite strength above 0, as --glp-alpha gives it; null for anything else.
pub fn strength(v: []const u8) ?f64 {
    const x = @import("fields.zig").pyFloat(v) orelse return null;
    if (!std.math.isFinite(x) or x <= 0) return null;
    return x;
}

/// Read the file the operator named and accept it; the message says why not, or null when it is read.
pub fn read(a: std.mem.Allocator, g: *?Vec) error{OutOfMemory}!?[]const u8 {
    const v = g.* orelse return null;
    const path = v.path orelse return "argument --glp-alpha: needs --glp";
    var why: glp.Why = .{};
    const loaded = load(a, path, v.alpha, &why) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        return try std.fmt.allocPrint(a, "argument --glp: {s}", .{why.text()});
    };
    g.* = .{ .path = path, .alpha = v.alpha, .vector = loaded.vector, .claimed = loaded.claimed };
    return null;
}

/// The file read whole: its vector and whether it claims a digest.
const Loaded = struct { vector: glp.Vector, claimed: bool };

fn load(a: std.mem.Allocator, path: []const u8, alpha: ?f64, why: *glp.Why) error{ BadGlpFile, OutOfMemory }!Loaded {
    // The strings the reader hands back borrow the file's bytes, so those bytes stay as long as the vector.
    const bytes = std.Io.Dir.cwd().readFileAlloc(startupIo(), path, a, .limited(file_limit)) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        why.set("cannot read '{s}': {s}", .{ path, reason(e) });
        return error.BadGlpFile;
    };
    const vector = glp.read(a, bytes, why) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        return error.BadGlpFile;
    };
    if (alpha == null and (!std.math.isFinite(vector.alpha) or vector.alpha <= 0)) {
        why.set("the file's own alpha_default is {d}; the strength must be above 0: pass --glp-alpha", .{vector.alpha});
        return error.BadGlpFile;
    }
    return .{ .vector = vector, .claimed = std.mem.indexOf(u8, bytes, digest_key) != null };
}

/// Why a file could not be read, in words the operator reads.
fn reason(e: anyerror) []const u8 {
    return switch (e) {
        error.FileNotFound => "no such file",
        error.AccessDenied => "permission denied",
        error.IsDir => "the path is a directory",
        error.BadPathName => "the path is not a file name",
        error.NameTooLong => "the path is too long",
        error.StreamTooLong => "the file is over the size limit",
        else => "the file cannot be read",
    };
}

/// The io a startup file read uses: the process's own, since the flags are read before the server takes one.
fn startupIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// The control vector the engine applies, after the checks that refuse one.
pub fn engine(a: std.mem.Allocator, dir: []const u8, model_type: []const u8, backend: []const u8, g: ?Vec) error{ OutOfMemory, GlpRefused }!?api.Glp {
    const v = g orelse return null;
    const vector = v.vector orelse return null;
    if (try refusal(a, dir, model_type, backend, v)) |why| {
        std.debug.print("tensorfold: {s}\n", .{why});
        return error.GlpRefused;
    }
    report(a, v);
    return api.Glp{ .vector = vector, .alpha = alphaOf(v) };
}

/// Why this checkpoint cannot take the vector, as the operator reads it; null: it can.
pub fn refusal(a: std.mem.Allocator, dir: []const u8, model_type: []const u8, backend: []const u8, v: Vec) error{OutOfMemory}!?[]const u8 {
    const vector = v.vector orelse return null;
    const model = modelOf(a, dir);
    var why: glp.Why = .{};
    if (model.hidden) |hidden| glp.checkModel(vector, hidden, model.repo, &why) catch {
        return try std.fmt.allocPrint(a, "--glp: {s}", .{why.text()});
    };
    if (isGemma(model_type)) return "--glp: Gemma's terms of use forbid circumventing its safety filters, so a Gemma checkpoint is never steered: serve it without --glp";
    if (!std.mem.eql(u8, model_type, steer_type)) return try std.fmt.allocPrint(a, "--glp: {s} checkpoints are not steered: only Qwen3.8-27B takes a control vector", .{model_type});
    if (model.hidden) |hidden| if (hidden != steer_hidden) return "--glp: this checkpoint is not Qwen3.8-27B: only the 5120-wide one takes a control vector";
    if (!metal(backend)) return try std.fmt.allocPrint(a, "--glp: only the Metal engine steers: this server runs the {s} engine", .{backendName(backend)});
    return null;
}

/// The checkpoint's hidden size and the repo it came from, as the vector's own check needs them.
const Model = struct {
    /// The model's hidden size, when its config names one.
    hidden: ?usize = null,
    /// The repo id the checkpoint came from ("Org/Name"), empty when it is not a cache snapshot.
    repo: []const u8 = "",
};

fn modelOf(a: std.mem.Allocator, dir: []const u8) Model {
    var m: Model = .{ .repo = cachedRepo(a, dir) };
    var problem: []const u8 = "";
    const bytes = (cfg.read(a, startupIo(), dir, &problem) catch null) orelse return m;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return m;
    if (doc != .object) return m;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    if (text.object.get("hidden_size")) |v| {
        if (v == .integer and v.integer > 0) m.hidden = @intCast(v.integer);
    }
    return m;
}

/// The repo id a cached checkpoint came from ("Org/Name"); empty when the folder is not a cache snapshot.
fn cachedRepo(a: std.mem.Allocator, dir: []const u8) []const u8 {
    const at = std.mem.lastIndexOf(u8, dir, "models--") orelse return "";
    const rest = dir[at + "models--".len ..];
    const sep = std.mem.indexOf(u8, rest, "/snapshots/") orelse return "";
    const owner = rest[0..sep];
    const dash = std.mem.indexOf(u8, owner, "--") orelse return "";
    return std.fmt.allocPrint(a, "{s}/{s}", .{ owner[0..dash], owner[dash + 2 ..] }) catch "";
}

/// Gemma's terms of use forbid circumventing its safety filters, so its checkpoints are never steered.
fn isGemma(model_type: []const u8) bool {
    return std.mem.startsWith(u8, model_type, "gemma");
}

/// The backend that steers: the Metal engine; the CUDA engines carry no control vector.
fn metal(backend: []const u8) bool {
    if (builtin.os.tag != .macos) return false;
    return std.mem.eql(u8, backend, "auto") or std.mem.eql(u8, backend, "mlx");
}

/// The engine a backend names, as the operator reads it.
fn backendName(backend: []const u8) []const u8 {
    if (std.mem.eql(u8, backend, "auto")) return if (builtin.os.tag == .macos) "Metal" else "CUDA";
    if (std.mem.eql(u8, backend, "mlx")) return "Metal";
    if (std.mem.eql(u8, backend, "cuda")) return "CUDA";
    if (std.mem.eql(u8, backend, "rocm")) return "ROCm";
    return backend;
}

/// The strength in force: the operator's, or the file's own.
pub fn alphaOf(v: Vec) f32 {
    if (v.alpha) |x| return @floatCast(x);
    return v.vector.?.alpha;
}

/// The line that reports what is in force: the file, the mode, the strength, the layers and where the vector came from.
pub fn report(a: std.mem.Allocator, v: Vec) void {
    const vector = v.vector orelse return;
    const sha = std.fmt.bytesToHex(vector.sha256, .lower);
    log.line("glp on: {s} (mode: {s}; alpha: {d:.2}; layers: {d} ({s}); base: {s}; digest: {s})", .{
        v.path orelse "",
        @tagName(vector.mode),
        alphaOf(v),
        vector.layers.len,
        layerIds(a, vector.layers),
        base(a, vector),
        fmt(a, "{s} ({s})", .{ sha[0..12], if (v.claimed) "verified" else "the file claims none" }),
    });
    if (vector.transferred) log.line("warning: this vector was derived at {s}, not at {s}: it is applied at the residual stream", .{ vector.derived_at, vector.hook_point });
}

/// The checkpoint the vector was derived from, with its revision when it names one.
fn base(a: std.mem.Allocator, v: glp.Vector) []const u8 {
    const model = if (v.base_model.len == 0) "an unnamed checkpoint" else v.base_model;
    if (v.base_revision.len == 0) return model;
    return fmt(a, "{s} @ {s}", .{ model, v.base_revision });
}

/// The layer ids a vector covers: "1-63" when they run and "1, 3, 5" when they do not.
fn layerIds(a: std.mem.Allocator, layers: []const glp.Layer) []const u8 {
    if (layers.len == 0) return "none";
    var run = true;
    for (layers, 0..) |l, i| if (i > 0 and l.id != layers[i - 1].id + 1) {
        run = false;
        break;
    };
    if (run) return fmt(a, "{d}-{d}", .{ layers[0].id, layers[layers.len - 1].id });
    var w: std.Io.Writer.Allocating = .init(a);
    for (layers, 0..) |l, i| {
        if (i > 0) w.writer.print(", ", .{}) catch {};
        w.writer.print("{d}", .{l.id}) catch {};
    }
    return w.toOwnedSlice() catch "?";
}

/// A formatted string, or a mark when the allocator says no.
fn fmt(a: std.mem.Allocator, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(a, f, args) catch "?";
}
