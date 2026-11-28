//! The control vector a serve starts with: the file it reads, the checks that refuse one, and the line that reports it.

const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const glp = api.glp;
const log = @import("log.zig");
const cfg = @import("checkpoint_config");
const cli = @import("cli.zig");

/// The flags the control vector adds to the server's own; a build that cannot apply one does not list them.
pub const flags = [_]cli.Flag{
    .{ .name = "--glp", .path = true, .native = true, .advertised = cli.steers() },
    .{ .name = "--glp-alpha", .native = true, .advertised = cli.steers() },
};

/// The model type of the family a residual vector steers: Qwen3.8-27B, on the Metal engine.
const steer_type = "qwen3_5";

/// The model type of the family a paired vector steers: DeepSeek-V4.1, whose EXL3 checkpoints the CUDA engine serves.
pub const deepseek_type = "deepseek_v41";

/// The hidden size of Qwen3.8-27B: it is 5120 wide.
const steer_hidden: usize = 5120;

/// The copies of the stream DeepSeek-V4.1 carries: a paired vector is applied to every one of them.
const hc_copies: usize = 4;

/// The model type of Inkling 2 Mini, whose Metal lane takes a residual vector one direction per decoder visit.
pub const inkling_type = "inkling2_loop";

/// Inkling 2 Mini's width, and the decoder visits a token makes: 24 layers, with layers 6-17 run three times.
const inkling_hidden: usize = 2304;
const inkling_visits: usize = 48;

/// What the --glp flags hold: the file the operator named, the strength and the vector the file carries.
pub const Vec = struct {
    /// The file's path, as the operator wrote it (null: no file named).
    path: ?[]const u8 = null,
    /// The strength the operator gave (null: the file's own alpha_default is in force).
    alpha: ?f32 = null,
    /// The vector the file carries, read once at startup (null: the file was not read).
    vector: ?glp.Vector = null,
};

/// The --glp file with the path the operator gave; null when the operator named none.
pub fn withPath(g: ?Vec, path: []const u8) Vec {
    return .{ .path = path, .alpha = if (g) |v| v.alpha else null };
}

/// The --glp file with the strength the operator gave; null when the operator named none.
pub fn withAlpha(g: ?Vec, alpha: f32) Vec {
    return .{ .path = if (g) |v| v.path else null, .alpha = alpha };
}

/// A strength from 0 as --glp-alpha gives it, 0 bypassing every edit; null for what FP32 cannot hold as itself.
pub fn strength(v: []const u8) ?f32 {
    return glp.strength(@import("fields.zig").pyFloat(v) orelse return null);
}

/// True when `name` is a control vector flag, now held in `g`; BadStrength for a --glp-alpha no FP32 strength holds.
pub fn flag(g: *?Vec, name: []const u8, v: []const u8) error{BadStrength}!bool {
    if (std.mem.eql(u8, name, "--glp")) {
        g.* = withPath(g.*, v);
    } else if (std.mem.eql(u8, name, "--glp-alpha")) {
        g.* = withAlpha(g.*, strength(v) orelse return error.BadStrength);
    } else return false;
    return true;
}

/// Read the file the operator named and accept it; the message says why not, or null when it is read.
pub fn read(a: std.mem.Allocator, g: *?Vec) error{OutOfMemory}!?[]const u8 {
    const v = g.* orelse return null;
    const path = v.path orelse return "argument --glp-alpha: needs --glp";
    var why: glp.Why = .{};
    // The strings the reader hands back borrow the file's bytes, so those bytes stay as long as the vector.
    const loaded = glp.load(a, startupIo(), path, v.alpha, &why) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        return try std.fmt.allocPrint(a, "argument --glp: {s}", .{why.text()});
    };
    g.* = .{ .path = path, .alpha = v.alpha, .vector = loaded.vector };
    return null;
}

/// The io a startup file read uses: the process's own, since the flags are read before the server takes one.
fn startupIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// The control vector the engine applies, after the checks that refuse one: `refusal` again for a caller without it.
pub fn engine(a: std.mem.Allocator, dir: []const u8, model_type: []const u8, backend: []const u8, g: ?Vec) error{ OutOfMemory, GlpRefused }!?api.glp.Glp {
    const v = g orelse return null;
    const vector = v.vector orelse return null;
    if (try refusal(a, dir, model_type, backend, v) != null) return error.GlpRefused;
    report(a, v);
    return api.glp.Glp{ .vector = vector, .alpha = alphaOf(v) };
}

/// Why this checkpoint cannot take the vector, as the operator reads it; null: it can.
pub fn refusal(a: std.mem.Allocator, dir: []const u8, model_type: []const u8, backend: []const u8, v: Vec) error{OutOfMemory}!?[]const u8 {
    const vector = v.vector orelse return null;
    const model = try modelOf(a, dir);
    if (isGemma(model_type)) return "--glp: Gemma checkpoints are never steered: serve it without --glp";
    if (std.mem.eql(u8, model_type, deepseek_type)) return pairedRefusal(a, backend, v, vector, model);
    if (std.mem.eql(u8, model_type, inkling_type)) return loopRefusal(a, backend, vector, model);
    if (!std.mem.eql(u8, model_type, steer_type)) {
        return try std.fmt.allocPrint(a, "--glp: {s} checkpoints are not steered: only Qwen3.8-27B, Inkling 2 Mini and DeepSeek-V4.1 take a control vector", .{model_type});
    }
    return residualRefusal(a, backend, v, vector, model);
}

/// Why an Inkling 2 Mini serve cannot take the vector: an FFN-write vector on the Metal engine, its ids decoder visits.
fn loopRefusal(a: std.mem.Allocator, backend: []const u8, vector: glp.Vector, model: Model) error{OutOfMemory}!?[]const u8 {
    if (!metal(backend)) return try std.fmt.allocPrint(a, "--glp: the {s} engine does not steer Inkling 2 Mini: only the Metal engine does", .{backendName(backend)});
    if (model.varloop) return "--glp: this checkpoint varies its loop (is_varloop): a vector's ids name the visits of three fixed passes";
    if (model.overflow) return "--glp: this checkpoint's loop_config counts more decoder visits than a visit id can name: the vector's ids cannot be checked";
    const hidden = model.hidden orelse return no_loop;
    const visits = model.visits orelse return no_loop;
    if (visits != inkling_visits) return try std.fmt.allocPrint(a, "--glp: this checkpoint's loop makes {d} decoder visits a token; the steered lane makes {d} (three core passes)", .{ visits, inkling_visits });
    var why: glp.Why = .{};
    glp.checkModel(vector, hidden, glp.ffn_site, model.repo, &why) catch return try because(a, &why);
    if (hidden != inkling_hidden) return "--glp: this checkpoint is not Inkling 2 Mini: only the 2304-wide one takes a control vector";
    for (vector.layers) |l| if (l.id >= visits) {
        return try std.fmt.allocPrint(a, "--glp: the vector steers decoder visit {d}; a token makes {d} visits, so its deepest is {d}", .{ l.id, visits, visits - 1 });
    };
    // The commit is compared only once the repo is known to be the vector's own.
    if (model.repo.len > 0 and vector.base_model.len > 0) glp.checkPairedRevision(vector, model.revision, &why) catch return try because(a, &why);
    return null;
}

const no_loop = "--glp: the checkpoint's config.json gives no text hidden size and loop_config, so the vector's visits cannot be checked";

/// The refusal as the operator reads it.
fn because(a: std.mem.Allocator, why: *const glp.Why) error{OutOfMemory}![]const u8 {
    return std.fmt.allocPrint(a, "--glp: {s}", .{why.text()});
}

/// Why a Qwen3.8-27B serve cannot take the vector: it takes a residual vector above strength 0, on the Metal engine.
fn residualRefusal(a: std.mem.Allocator, backend: []const u8, v: Vec, vector: glp.Vector, model: Model) error{OutOfMemory}!?[]const u8 {
    var why: glp.Why = .{};
    glp.checkModel(vector, model.hidden orelse vector.width, glp.residual_site, model.repo, &why) catch return try because(a, &why);
    if (model.layers) |layers| {
        for (vector.layers) |l| {
            if (l.id >= layers) return try std.fmt.allocPrint(a, "--glp: the vector steers layer {d} of a model with {d} layers: its deepest layer is {d}", .{ l.id, layers, layers - 1 });
        }
    }
    if (model.hidden) |hidden| if (hidden != steer_hidden) return "--glp: this checkpoint is not Qwen3.8-27B: only the 5120-wide one takes a control vector";
    if (!metal(backend)) return try std.fmt.allocPrint(a, "--glp: only the Metal engine steers: this server runs the {s} engine", .{backendName(backend)});
    if (alphaOf(v) == 0) return "--glp: Qwen3.8-27B takes a strength above 0; the zero that bypasses every edit is for DeepSeek-V4.1: pass --glp-alpha above 0";
    return null;
}

/// Why a DeepSeek-V4.1 serve cannot take the vector: it takes a paired vector on the CUDA engine, as its config fits.
fn pairedRefusal(a: std.mem.Allocator, backend: []const u8, v: Vec, vector: glp.Vector, model: Model) error{OutOfMemory}!?[]const u8 {
    if (!cuda(backend)) {
        return try std.fmt.allocPrint(a, "--glp: the {s} engine does not steer DeepSeek-V4.1: only the CUDA engine does", .{backendName(backend)});
    }
    const hidden = model.hidden orelse return no_dims;
    const layers = model.layers orelse return no_dims;
    if (model.copies) |copies| if (copies != hc_copies) {
        return try std.fmt.allocPrint(a, "--glp: the vector is applied to {d} stream copies; this checkpoint carries {d}", .{ hc_copies, copies });
    };
    var why: glp.Why = .{};
    glp.checkPairedModel(vector, hidden, layers, model.repo, &why) catch return try because(a, &why);
    // The commit is compared only once the repo is known to be the vector's own.
    if (model.repo.len > 0 and vector.base_model.len > 0) glp.checkPairedRevision(vector, model.revision, &why) catch return try because(a, &why);
    glp.checkStrength(vector, alphaOf(v), &why) catch return try because(a, &why);
    return null;
}

const no_dims = "--glp: the checkpoint's config.json gives no text hidden size and depth, so the vector cannot be checked against it";

/// The checkpoint's hidden size, depth, stream copies, and the repo and commit it came from, for the vector's check.
const Model = struct {
    /// The model's hidden size, when its config names one.
    hidden: ?usize = null,
    /// The model's number of backbone blocks, when its config names one.
    layers: ?usize = null,
    /// The copies of the stream (hc_mult), when its config names them.
    copies: ?usize = null,
    /// The decoder visits a token makes: the layers, plus loop_config's repeated core passes.
    visits: ?usize = null,
    /// Whether loop_config varies its passes (is_varloop).
    varloop: bool = false,
    /// Whether loop_config's passes multiply past the visits a vector's ids can name.
    overflow: bool = false,
    /// The repo id the checkpoint came from ("Org/Name"), empty when it is not a cache snapshot.
    repo: []const u8 = "",
    /// The commit a cache snapshot is, empty when it is not one.
    revision: []const u8 = "",
};

fn modelOf(a: std.mem.Allocator, dir: []const u8) error{OutOfMemory}!Model {
    var m: Model = .{};
    if (try glp.checkpoint.source(a, dir)) |source| {
        m.repo = source.repo;
        m.revision = source.revision;
    }
    var problem: []const u8 = "";
    const bytes = (cfg.read(a, startupIo(), dir, &problem) catch null) orelse return m;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return m;
    if (doc != .object) return m;
    // The text model's numbers live in text_config; a vision tower's sit beside it and are never read.
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    m.hidden = positive(text.object.get("hidden_size"));
    m.layers = positive(text.object.get("num_hidden_layers") orelse doc.object.get("num_hidden_layers"));
    m.copies = positive(text.object.get("hc_mult") orelse doc.object.get("hc_mult"));
    if (text.object.get("loop_config")) |loop| if (loop == .object) {
        const passes = positive(loop.object.get("num_passes"));
        const start = positive(loop.object.get("core_start"));
        const end = positive(loop.object.get("core_end"));
        if (m.layers != null and passes != null and start != null and end != null and end.? > start.?) {
            // a config that counts past the visit ids a vector may name is refused, never wrapped
            const core = std.math.mul(usize, passes.? - 1, end.? - start.?) catch {
                m.overflow = true;
                return m;
            };
            m.visits = std.math.add(usize, m.layers.?, core) catch {
                m.overflow = true;
                return m;
            };
        }
        if (loop.object.get("is_varloop")) |vl| m.varloop = vl != .bool or vl.bool;
    };
    return m;
}

/// A positive integer a config holds, null when the key is absent or holds anything else.
fn positive(value: ?std.json.Value) ?usize {
    const v = value orelse return null;
    return if (v == .integer and v.integer > 0) @intCast(v.integer) else null;
}

/// Gemma checkpoints are never steered.
fn isGemma(model_type: []const u8) bool {
    return std.mem.startsWith(u8, model_type, "gemma");
}

/// The backend that steers Qwen: the Metal engine.
fn metal(backend: []const u8) bool {
    if (builtin.os.tag != .macos) return false;
    return std.mem.eql(u8, backend, "auto") or std.mem.eql(u8, backend, "mlx");
}

/// The backend that steers DeepSeek-V4.1: the CUDA engine, which `auto` names on Linux; Metal and ROCm carry no vector.
fn cuda(backend: []const u8) bool {
    return std.mem.eql(u8, backend, "cuda") or (builtin.os.tag == .linux and std.mem.eql(u8, backend, "auto"));
}

/// The engine a backend names, as the operator reads it.
fn backendName(backend: []const u8) []const u8 {
    if (std.mem.eql(u8, backend, "auto")) return if (builtin.os.tag == .macos) "Metal" else "CUDA";
    if (std.mem.eql(u8, backend, "mlx")) return "Metal";
    if (std.mem.eql(u8, backend, "cuda")) return "CUDA";
    if (std.mem.eql(u8, backend, "rocm")) return "ROCm";
    return backend;
}

/// The strength in force: the operator's, or the file's own; a negative zero is the zero it equals.
pub fn alphaOf(v: Vec) f32 {
    const x = v.alpha orelse v.vector.?.alpha;
    return if (x == 0) 0 else x;
}

/// The line that reports what is in force: the file (escaped), its SHA-256, that replies are steered, and the rest.
pub fn report(a: std.mem.Allocator, v: Vec) void {
    const vector = v.vector orelse return;
    const sha = std.fmt.bytesToHex(vector.sha256, .lower);
    const file_sha = std.fmt.bytesToHex(vector.file_sha256, .lower);
    const digest = fmt(a, "{s} ({s})", .{ sha[0..12], if (vector.digest_claimed) "verified" else "the file claims none" });
    if (vector.isPaired()) {
        const alpha = alphaOf(v);
        const verdict = if (alpha == 0) "glp loaded but inactive: alpha 0 bypasses every edit and replies are unsteered" else "glp on: replies are steered";
        log.line("{s} (add vector in \"{f}\"; hook: {s}; spec v{d}; file sha256: {s}; alpha: {d}; layers: {d} ({s}); base: {s}; tensor digest: {s})", .{
            verdict,
            std.zig.fmtString(v.path orelse ""),
            vector.hook_point,
            vector.spec_version,
            file_sha[0..],
            alpha,
            vector.layers.len,
            layerIds(a, vector.layers),
            base(a, vector),
            digest,
        });
    } else {
        log.line("{s} the control vector in \"{f}\" (file sha256: {s}; alpha: {d:.2}; layers: {d} ({s}); base: {s}; tensor digest: {s})", .{
            if (alphaOf(v) == 0) "glp loaded but inactive: alpha 0 bypasses every edit and replies are unsteered; it is" else "glp on: replies are steered by",
            std.zig.fmtString(v.path orelse ""),
            file_sha[0..],
            alphaOf(v),
            vector.layers.len,
            layerIds(a, vector.layers),
            base(a, vector),
            digest,
        });
    }
    const applied = appliedSite(vector.hook_point);
    if (vector.transferred) log.line("warning: this vector was derived at {f}, not at {f}: it is applied at {s}", .{ std.zig.fmtString(vector.derived_at), std.zig.fmtString(vector.hook_point), applied });
}

/// The site the lane applies the vector at, as the operator reads it: the file's hook, or the residual stream's name.
fn appliedSite(hook: []const u8) []const u8 {
    return if (std.mem.eql(u8, hook, "residual_stream") or std.mem.eql(u8, hook, glp.residual_site)) "the residual stream" else hook;
}

/// The checkpoint the vector was derived from, with its revision when it names one; the file's own words, escaped.
fn base(a: std.mem.Allocator, v: glp.Vector) []const u8 {
    if (v.base_model.len == 0) return "an unnamed checkpoint";
    if (v.base_revision.len == 0) return fmt(a, "{f}", .{std.zig.fmtString(v.base_model)});
    return fmt(a, "{f} @ {f}", .{ std.zig.fmtString(v.base_model), std.zig.fmtString(v.base_revision) });
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
