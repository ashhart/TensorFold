//! Nemotron-H admission on HIP before anything loads: the format, every tensor, what is refused, a rank's bytes.
const std = @import("std");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const plan = @import("plan.zig");
const widths = @import("widths.zig");

const Config = config.Config;

/// The MLX affine checkpoints the HIP engine reads: 4-, 5-, 6- or 8-bit codes in groups of 64, a width a matrix.
pub const formats = widths.formats;
/// Device bytes a rank keeps for the forward's scratch until the engine counts its own.
pub const scratch_bytes: u64 = 2 << 30;
/// Rows of the vocabulary head the MTP head drafts among (draft_ids.txt).
pub const draft_rows = 49152;

/// What admission reads from the model folder.
pub const Facts = struct {
    /// config.json's text.
    config: []const u8,
    /// model.safetensors.index.json's text.
    index: []const u8,
    /// The tensor names of mtp-4bit.safetensors; null when the file is not there (no drafts).
    mtp: ?[]const []const u8 = null,
    /// The quantization the MTP head's file states in its metadata; null: none stated (the stock 4-bit head).
    mtp_quantization: ?[]const u8 = null,
    /// A living-weights file (a lesson Sliding Weights saved) sits beside the shards.
    living: bool = false,
};

/// The rank asking, the window it serves and the bytes its card grants.
pub const Ask = struct { world: u32 = 1, rank: u32 = 0, window: u64, room: u64, act: u64 = 2 };

pub const Verdict = struct {
    config: Config,
    mtp: bool,
    /// The MTP head's own width, as its file states.
    head: config.Width = .{},
    weights: u64,
    token: u64,
    stream: u64,
    /// The longest window one stream may hold beside the weights and scratch.
    most: u64,
    fits: bool,
};

pub const Refused = error{Refused};

fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

pub fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// Names checked off against a set: the first one missing, and how many were found.
const Check = struct {
    have: *const std.StringHashMapUnmanaged(void),
    missing: ?[]const u8 = null,
    buf: [192]u8 = undefined,
    found: usize = 0,

    fn add(self: *Check, name: []const u8) anyerror!void {
        if (self.have.contains(name)) {
            self.found += 1;
        } else if (self.missing == null) {
            @memcpy(self.buf[0..name.len], name);
            self.missing = self.buf[0..name.len];
        }
    }
};

/// A set of names being filled.
const Set = struct {
    a: std.mem.Allocator,
    m: *std.StringHashMapUnmanaged(void),

    fn add(self: Set, name: []const u8) anyerror!void {
        try self.m.put(self.a, try self.a.dupe(u8, name), {});
    }
};

/// A tensor name `have` holds that the checkpoint (or the MTP head when `mtp`) does not list, or null.
fn extra(
    a: std.mem.Allocator,
    have: *const std.StringHashMapUnmanaged(void),
    c: *const Config,
    mtp: bool,
) !?[]const u8 {
    var want: std.StringHashMapUnmanaged(void) = .empty;
    const set: Set = .{ .a = a, .m = &want };
    if (mtp) try tensors.eachMtp(set, Set.add) else try tensors.eachText(c, set, Set.add);
    var it = have.keyIterator();
    while (it.next()) |k| if (!want.contains(k.*)) return k.*;
    return null;
}

/// Why `missing` is absent: a learned bf16 projection (its words without tables) reads as Living Weights.
fn missingWhy(
    a: std.mem.Allocator,
    have: *const std.StringHashMapUnmanaged(void),
    missing: []const u8,
    where: []const u8,
    problem: *[]const u8,
) Refused {
    if (std.mem.endsWith(u8, missing, ".scales")) {
        const stem = missing[0 .. missing.len - ".scales".len];
        const words = std.fmt.allocPrint(a, "{s}.weight", .{stem}) catch return error.Refused;
        if (have.contains(words)) return refuse(
            a,
            problem,
            "{s} is stored unquantized (a projection Living Weights taught): Nemotron on HIP reads " ++
                widths.served_text ++ " alone for now",
            .{stem},
        );
    }
    return refuse(a, problem, "{s} lacks the Nemotron-H tensor {s}", .{ where, missing });
}

/// Checks the names in `have` are exactly those the text checkpoint (or the MTP head when `mtp`) holds.
fn names(
    a: std.mem.Allocator,
    have: *const std.StringHashMapUnmanaged(void),
    c: *const Config,
    mtp: bool,
    where: []const u8,
    problem: *[]const u8,
) Refused!void {
    var check: Check = .{ .have = have };
    const walked = if (mtp) tensors.eachMtp(&check, Check.add) else tensors.eachText(c, &check, Check.add);
    walked catch |err| return refuse(a, problem, "{s}: Nemotron-H's tensor names do not fit ({t})", .{ where, err });
    if (check.missing) |m| return missingWhy(a, have, m, where, problem);
    if (check.found != have.count()) {
        const name = (extra(a, have, c, mtp) catch return error.Refused) orelse "?";
        if (!mtp and std.mem.startsWith(u8, name, "mtp.")) return refuse(
            a,
            problem,
            "{s} holds the MTP head ({s}) inside the model's shards: Nemotron on HIP reads the head from {s} " ++
                "beside them",
            .{ where, name, tensors.mtp_file },
        );
        return refuse(a, problem, "{s} holds {s}, which Nemotron-H on HIP does not read", .{ where, name });
    }
}

/// The index's weight map as a set of names.
fn indexNames(a: std.mem.Allocator, text: []const u8) !std.StringHashMapUnmanaged(void) {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (doc != .object) return error.BadIndex;
    const map = doc.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    var set: std.StringHashMapUnmanaged(void) = .empty;
    for (map.object.keys()) |k| try set.put(a, k, {});
    return set;
}

/// The checkpoint's format and what it holds, refused unless the HIP engine reads all of it.
pub fn checkpoint(a: std.mem.Allocator, f: Facts, problem: *[]const u8) Refused!Config {
    var why: config.Why = .{};
    var c = config.parseWith(a, f.config, &why, .{ .bf16_embed = true }) catch |err| {
        if (why.len > 0) return refuse(a, problem, "{s}", .{why.text()});
        return refuse(a, problem, "config.json is not a Nemotron-H config the HIP engine reads ({t})", .{err});
    };
    if (c.format == .modelopt) return refuse(
        a,
        problem,
        "this is a ModelOpt (FP8 and NVFP4) checkpoint: the CUDA engine serves it; Nemotron on HIP reads " ++
            "MLX affine checkpoints",
        .{},
    );
    var name: [96]u8 = undefined;
    if (widths.unserved(&c, &name)) |m| return refuse(
        a,
        problem,
        "{s} is {d}-bit in groups of {d}: Nemotron on HIP reads " ++ widths.served_text,
        .{ m.name, m.width.bits, m.width.group },
    );
    if (c.pocket > 0) return refuse(
        a,
        problem,
        "config.json adds {d} pocket neurons to the shared experts; Nemotron on HIP does not serve pocket neurons yet",
        .{c.pocket},
    );
    if (f.living) return refuse(
        a,
        problem,
        "living-weights files sit beside the shards: Nemotron on HIP does not serve Living Weights yet; " ++
            "serve a folder without them",
        .{},
    );
    var have = indexNames(a, f.index) catch |err| return refuse(
        a,
        problem,
        "model.safetensors.index.json is unreadable ({t})",
        .{err},
    );
    // as MLX loads it, a table without scales is not quantized, whatever config.json lists for it
    if (tensors.hasBf16Embed(have.contains(tensors.embed ++ ".weight"), have.contains(tensors.embed ++ ".scales"))) c.embed_bf16 = true;
    try names(a, &have, &c, false, "the checkpoint's index", problem);
    if (f.mtp) |list| {
        var head: std.StringHashMapUnmanaged(void) = .empty;
        for (list) |n| head.put(a, n, {}) catch return error.Refused;
        try names(a, &head, &c, true, tensors.mtp_file, problem);
    }
    return c;
}

/// The MTP head's width from its file's metadata, refused by name unless the launches read it.
fn headWidth(a: std.mem.Allocator, f: Facts, problem: *[]const u8) Refused!config.Width {
    const w = widths.headOf(a, f.mtp_quantization) catch return refuse(
        a,
        problem,
        "{s} states its quantization as {s}: Nemotron on HIP reads the head as MLX affine bits and group_size",
        .{ tensors.mtp_file, f.mtp_quantization orelse "" },
    );
    if (!widths.served(w)) return refuse(
        a,
        problem,
        "{s} is {d}-bit in groups of {d}: Nemotron on HIP reads " ++ widths.served_text,
        .{ tensors.mtp_file, w.bits, w.group },
    );
    return w;
}

/// Admits the checkpoint of `f` on rank `ask.rank` of `ask.world` at `ask.window` positions in `ask.room` bytes.
pub fn admit(a: std.mem.Allocator, f: Facts, ask: Ask, problem: *[]const u8) Refused!Verdict {
    const c = try checkpoint(a, f, problem);
    const mtp = f.mtp != null;
    const head = try headWidth(a, f, problem);
    const want: plan.Head = .{ .mtp = mtp, .drafts = if (mtp) draft_rows else 0, .width = head };
    const w = plan.weights(&c, ask.world, .device, want) catch
        return refuse(
            a,
            problem,
            "--tp {d} does not split Nemotron-H's Mamba groups, heads, experts and vocabulary evenly; " ++
                "choose 1, 2, 4 or 8",
            .{ask.world},
        );
    const token = plan.tokenBytes(&c, ask.world, ask.act, mtp) catch return refuse(
        a,
        problem,
        "--tp {d} does not split Nemotron-H's kv heads",
        .{ask.world},
    );
    const stream = plan.streamBytes(&c, ask.world, ask.act) catch return refuse(
        a,
        problem,
        "--tp {d} does not split Nemotron-H's Mamba heads",
        .{ask.world},
    );
    const bytes = w.device(ask.rank);
    const most = plan.window(ask.room, bytes, scratch_bytes, token, stream);
    return .{
        .config = c,
        .mtp = mtp,
        .head = head,
        .weights = bytes,
        .token = token,
        .stream = stream,
        .most = most,
        .fits = most >= ask.window,
    };
}

/// The refusal of a verdict that does not fit, naming what the card grants.
pub fn tooBig(a: std.mem.Allocator, v: Verdict, ask: Ask, problem: *[]const u8) Refused {
    return refuse(
        a,
        problem,
        "Nemotron on {d} rank(s) holds {d:.2} GiB of weights a rank and {d:.1} GiB of scratch, of the " ++
            "{d:.2} GiB this card grants: a {d}-token window does not fit (at most {d}); lower --context or " ++
            "add ranks (--tp)",
        .{ ask.world, gibs(v.weights), gibs(scratch_bytes), gibs(ask.room), ask.window, v.most },
    );
}

test {
    _ = @import("admit_test.zig");
}
