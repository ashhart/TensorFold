//! The execution witness's Qwen3.8-27B-class Metal path: the 9B and the 27B, MLX affine 4-bit, text.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const q = @import("tensorfold").qwen27;
const q27_eng = @import("qwen27_engine.zig");
const disk_identity = @import("disk_identity.zig");
const weights_witness = @import("weights_witness.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;
const field = weights_witness.field;

/// The path of a qualified geometry (`q.model.supported`), every module affine 4-bit in groups of 64; else null.
pub fn pathOf(c: q.config.Config, widths_tag: u64) ?[]const u8 {
    if (widths_tag != 0 or c.looped or c.tied_head or c.ranks != 1 or c.vocab != 248320) return null;
    if (c.experts == 0 and c.hidden == 4096 and c.layers == 32) return "qwen3_5-9b/metal/tp1/mlx-q4g64/text";
    if (c.experts == 0 and c.hidden == 5120 and c.layers == 64) return "qwen3_5-27b/metal/tp1/mlx-q4g64/text";
    return null;
}

/// What the path binds beside the weights: the libraries the load built, the GPU, the profile and the served shape.
pub const Qwen27 = struct {
    pub const domain = "tensorfold qwen27 metal engine witness v1";
    /// Every library the load built, each with its compile options (`mtl.built`).
    kernels: [32]u8 = @splat(0),
    /// The chip, hashed with the OS build: the OS's compiler builds the GPU code.
    device: []const u8 = "",
    tensor_units: bool = false,
    profile: disk_identity.Profile = .{ .family = "qwen3_5", .backend = "metal" },
    /// Context, streams, a pass's prompt rows, verify window, KV pages, K/V growth step, copies and the lone path.
    words: [8]u64 = @splat(0),
    /// The load's library builds, watched from before the weights load until `bind`.
    built: ?mtl.built.Built = null,

    /// Nothing beyond the checkpoint: the kernels are source text in the binary, built at load.
    pub fn stamp(_: *Qwen27, _: Allocator, _: std.Io, _: *std.ArrayList([]const u8)) !void {}

    pub fn keep(b: *const Qwen27, h: *Sha256, _: Allocator, _: std.Io) !void {
        field(h, &b.kernels);
        disk_identity.machine(h, b.device);
        field(h, &.{@intFromBool(b.tensor_units)});
        disk_identity.profile(h, b.profile);
    }

    pub fn tail(b: *const Qwen27) []const u8 {
        return std.mem.sliceAsBytes(&b.words);
    }
};

pub const Witness = weights_witness.Witness(Qwen27);

/// Whether `o` may take the path before the weights load: the flag, one Mac, text, nothing learned or drafted.
pub fn wanted(o: api.Open) bool {
    return weights_witness.wanted(o) and !q27_eng.isSenseNova(o.model_type);
}

/// Before the load: stamps every weight file and watches the libraries the load builds; null when `o` is off the path.
pub fn begin(gpa: Allocator, io: std.Io, o: api.Open) error{OutOfMemory}!?*Witness {
    if (!wanted(o)) return null;
    const w = try Witness.openOrFailed(gpa, io, o.dir, .{ .profile = .{ .family = o.model_type, .backend = "metal" } });
    if (w.stamps.len == 0) return w; // hash_failed already: nothing to watch
    w.backend.built = .init(w.arena.allocator());
    _ = w.backend.built.?.watch(); // refused while another load is watched: then void, and `bind` leaves the path
    return w;
}

/// Stops watching builds, after any report in flight; null when the record is not whole or the watch was lost.
fn stopWatching(w: *Witness) ?[32]u8 {
    const b = if (w.backend.built) |*b| b else return null;
    return b.stop();
}

/// The facts the host gives once loaded, given directly.
pub fn set(w: *Witness, path: []const u8, kernel_digest: [32]u8, device: []const u8, tensor_units: bool, words: [8]u64) void {
    w.path = path;
    w.backend.kernels = kernel_digest;
    w.backend.device = device;
    w.backend.tensor_units = tensor_units;
    w.backend.words = words;
}

/// After the load: true when host `h` (qwen27_served.zig) serves the path as loaded; the witness then holds its facts.
pub fn bind(w: *Witness, h: anytype) bool {
    const kernels: [32]u8 = if (w.backend.built == null) @splat(0) else stopWatching(w) orelse return false;
    const m = h.model;
    if (m.ane != null or m.idle_release != null or h.draft != null or h.mtp != null or h.sn_mtp != null or h.vision != null) return false;
    const path = pathOf(m.config, m.widths_tag) orelse return false;
    const name = w.arena.allocator().dupe(u8, std.mem.span(m.device.name())) catch return false;
    const paged = q.decode_round.pagesOn() catch return false;
    const words = [8]u64{ h.host.info_.context_window, h.runner.slots, h.back.chunk, h.back.window, @intFromBool(paged), h.runner.kv_step orelse 0, @intFromBool(h.back.copies), @intFromBool(h.host.lone != null) };
    set(w, path, kernels, name, m.device.tensorUnits(), words);
    return true;
}

/// A loaded host and the witness it took onto the path, closed together.
const Served = struct {
    gpa: Allocator,
    host: *anyopaque,
    closeHost: *const fn (*anyopaque) void,
    witness: *Witness,

    fn close(ctx: *anyopaque) void {
        const s: *Served = @ptrCast(@alignCast(ctx));
        s.witness.close();
        s.closeHost(s.host);
        s.gpa.destroy(s);
    }
};

/// The engine to serve: with the witness's facts when it is on the path, else `h` alone (unsupported_path).
pub fn serve(gpa: Allocator, h: anytype, closeHost: *const fn (*anyopaque) void, witness: ?*Witness) error{OutOfMemory}!api.Opened {
    const plain: api.Opened = .{ .engine = h.engine(), .close = closeHost, .ctx = h };
    const w = witness orelse return plain;
    if (!bind(w, h)) return plain;
    const s = try gpa.create(Served);
    s.* = .{ .gpa = gpa, .host = h, .closeHost = closeHost, .witness = w };
    h.host.info_.witness = w.source();
    w.adopted = true;
    return .{ .engine = h.engine(), .close = Served.close, .ctx = s };
}

/// A witness the host did not adopt (a refusal, a failure or another path), its watch stopped and freed.
pub fn release(w: ?*Witness) void {
    const x = w orelse return;
    if (x.adopted) return;
    _ = stopWatching(x);
    x.close();
}

test {
    _ = @import("qwen27_witness_test.zig");
}
