//! Nemotron-H admission on HIP before anything loads: the format, every tensor, what is refused, a rank's bytes.
const std = @import("std");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const plan = @import("plan.zig");

const Config = config.Config;

/// The MLX affine checkpoints the HIP engine reads: 4-bit codes in groups of 64.
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
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
    /// A living-weights file (a lesson Sliding Weights saved) sits beside the shards.
    living: bool = false,
};

/// The rank asking, the window it serves and the bytes its card grants.
pub const Ask = struct { world: u32 = 1, rank: u32 = 0, window: u64, room: u64, act: u64 = 2 };

pub const Verdict = struct {
    config: Config,
    mtp: bool,
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
            "{s} is stored unquantized (a projection Living Weights taught): Nemotron on HIP reads MLX " ++
                "4-bit projections alone for now",
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
    const c = config.parse(a, f.config, &why) catch |err| {
        if (why.len > 0) return refuse(a, problem, "{s}", .{why.text()});
        return refuse(a, problem, "config.json is not a Nemotron-H config the HIP engine reads ({t})", .{err});
    };
    if (c.format == .modelopt) return refuse(
        a,
        problem,
        "this is a ModelOpt (FP8 and NVFP4) checkpoint: the CUDA engine serves it; Nemotron on HIP reads " ++
            "MLX affine 4-bit checkpoints",
        .{},
    );
    if (c.bits != 4 or c.group_size != 64) return refuse(
        a,
        problem,
        "Nemotron on HIP reads MLX affine 4-bit groups of 64; this checkpoint is {d}-bit in groups of {d}",
        .{ c.bits, c.group_size },
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
    try names(a, &have, &c, false, "the checkpoint's index", problem);
    if (f.mtp) |list| {
        var head: std.StringHashMapUnmanaged(void) = .empty;
        for (list) |n| head.put(a, n, {}) catch return error.Refused;
        try names(a, &head, &c, true, tensors.mtp_file, problem);
    }
    return c;
}

/// Admits the checkpoint of `f` on rank `ask.rank` of `ask.world` at `ask.window` positions in `ask.room` bytes.
pub fn admit(a: std.mem.Allocator, f: Facts, ask: Ask, problem: *[]const u8) Refused!Verdict {
    const c = try checkpoint(a, f, problem);
    const mtp = f.mtp != null;
    const w = plan.weights(&c, ask.world, .device, .{ .mtp = mtp, .drafts = if (mtp) draft_rows else 0 }) catch
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
        "Nemotron 4-bit on {d} rank(s) holds {d:.2} GiB of weights a rank and {d:.1} GiB of scratch, of the " ++
            "{d:.2} GiB this card grants: a {d}-token window does not fit (at most {d}); lower --context or " ++
            "add ranks (--tp)",
        .{ ask.world, gibs(v.weights), gibs(scratch_bytes), gibs(ask.room), ask.window, v.most },
    );
}

const testing = std.testing;
const stock_config = @embedFile("../fixtures/config.json");
const stock_index = @embedFile("../fixtures/index.json");
const gib: u64 = 1 << 30;

/// The stock MTP file's names, from the fixture's lines.
fn stockMtp(a: std.mem.Allocator) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, @embedFile("../fixtures/mtp_names.txt"), "\n\r ");
    while (it.next()) |n| try list.append(a, n);
    return list.items;
}

/// `text` with every `from` replaced by `to`.
fn edit(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]const u8 {
    try testing.expect(std.mem.indexOf(u8, text, from) != null);
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// admit() must refuse with a reason naming each of `words`.
fn expectRefused(a: std.mem.Allocator, f: Facts, ask: Ask, words: []const []const u8) !void {
    var problem: []const u8 = "";
    try testing.expectError(error.Refused, admit(a, f, ask, &problem));
    for (words) |w| if (std.mem.indexOf(u8, problem, w) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ problem, w });
        return error.TestUnexpectedResult;
    };
}

test "the stock checkpoint and its MTP head are admitted; on one V620 a stream takes the model's window" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const f: Facts = .{ .config = stock_config, .index = stock_index, .mtp = try stockMtp(a) };
    const one = try admit(a, f, .{ .window = 262144, .room = 30 * gib }, &problem);
    try testing.expect(one.fits and one.mtp);
    try testing.expectEqual(@as(usize, 52), one.config.layers);
    const two = try admit(a, f, .{ .world = 2, .rank = 1, .window = 262144, .room = 30 * gib }, &problem);
    try testing.expect(two.fits and two.weights < one.weights / 2);
    const lone = try admit(
        a,
        .{ .config = stock_config, .index = stock_index },
        .{ .window = 4096, .room = 30 * gib },
        &problem,
    );
    try testing.expect(!lone.mtp and lone.weights < one.weights);
    const small = try admit(a, f, .{ .window = 4096, .room = 16 * gib }, &problem);
    try testing.expect(!small.fits);
    try testing.expectEqual(error.Refused, tooBig(a, small, .{ .window = 4096, .room = 16 * gib }, &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "16.00 GiB this card grants") != null);
}

test "altered checkpoints are refused with a reason" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ask: Ask = .{ .window = 4096, .room = 30 * gib };
    const mtp = try stockMtp(a);
    const gone = "backbone.layers.7.mixer.A_log";
    try expectRefused(
        a,
        .{
            .config = stock_config,
            .index = try edit(a, stock_index, "\"" ++ gone ++ "\"", "\"backbone.layers.7.mixer.B_log\""),
        },
        ask,
        &.{gone},
    );
    const scales = "backbone.layers.12.mixer.o_proj.scales";
    try expectRefused(
        a,
        .{ .config = stock_config, .index = try edit(a, stock_index, "\"" ++ scales ++ "\"", "\"x\"") },
        ask,
        &.{ "backbone.layers.12.mixer.o_proj", "Living Weights" },
    );
    try expectRefused(
        a,
        .{
            .config = stock_config,
            .index = try edit(a, stock_index, "\"lm_head.weight\"", "\"lm_head.weight\": \"a\", \"vision.w\""),
        },
        ask,
        &.{"vision.w"},
    );
    try expectRefused(
        a,
        .{ .config = try edit(a, stock_config, "\"bits\": 4", "\"bits\": 8"), .index = stock_index },
        ask,
        &.{"8-bit in groups of 64"},
    );
    try expectRefused(
        a,
        .{ .config = try edit(a, stock_config, "\"group_size\": 64", "\"group_size\": 32"), .index = stock_index },
        ask,
        &.{"groups of 32"},
    );
    try expectRefused(
        a,
        .{
            .config = try edit(
                a,
                stock_config,
                "\"quantization\": {",
                "\"quantization\": {\"quant_method\": \"modelopt\",",
            ),
            .index = stock_index,
        },
        ask,
        &.{ "ModelOpt", "CUDA" },
    );
    try expectRefused(
        a,
        .{ .config = try edit(a, stock_config, "\"mode\": \"affine\"", "\"mode\": \"nvfp4\""), .index = stock_index },
        ask,
        &.{"nvfp4"},
    );
    try expectRefused(
        a,
        .{
            .config = try edit(
                a,
                stock_config,
                "\"model_type\": \"nemotron_h\"",
                "\"model_type\": \"nemotron_h\", \"tensorfold_learned_neurons\": 64",
            ),
            .index = stock_index,
        },
        ask,
        &.{"pocket"},
    );
    try expectRefused(a, .{ .config = stock_config, .index = stock_index, .living = true }, ask, &.{"Living Weights"});
    try expectRefused(a, .{ .config = stock_config, .index = stock_index, .mtp = mtp[1..] }, ask, &.{tensors.mtp_file});
    try expectRefused(
        a,
        .{ .config = stock_config, .index = stock_index },
        .{ .world = 3, .window = 4096, .room = 30 * gib },
        &.{"--tp 3"},
    );
}
