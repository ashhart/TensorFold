//! What a run may use, resolved once at open; nothing below the engine reads the environment, it reads this.

const std = @import("std");
const Launcher = @import("launches.zig").Launcher;
const affine = @import("launches/affine.zig");

pub const Choice = enum { auto, on, off };
/// The prefill attention tile: `f32` keeps the one-row kernel.
pub const Attention = enum { auto, f32 };
/// One kernel family switched to its reference, apart from `kernels=reference` that switches them all.
pub const Pick = enum { auto, reference };

/// MTP drafting: `--mtp-drafts` and `--mtp-confidence`.
pub const Mtp = struct { drafts: u8 = 3, confidence: f64 = 0.3 };

/// A path a rank reads locally (it never travels with the policy).
pub const Path = struct {
    buf: [160]u8 = @splat(0),

    pub fn slice(p: *const Path) ?[]const u8 {
        const n = std.mem.indexOfScalar(u8, &p.buf, 0) orelse p.buf.len;
        return if (n == 0) null else p.buf[0..n];
    }

    fn set(p: *Path, text: []const u8) error{BadValue}!void {
        if (text.len >= p.buf.len) return error.BadValue;
        p.buf = @splat(0);
        @memcpy(p.buf[0..text.len], text);
    }
};

pub const Policy = struct {
    matrix: Choice = .auto,
    attention: Attention = .auto,
    kernels: Pick = .auto,
    graphs: Choice = .auto,
    /// Reference kernels one at a time: decode stream tile, prefill GEMM tile, fused decode tails, chunked recurrence.
    stream: Pick = .auto,
    gemm: Pick = .auto,
    fuse: Pick = .auto,
    gdn: Pick = .auto,
    prefill_step: u32 = 1024,
    /// Set from the serve flags, not keys: rank 0's travel to the other ranks.
    mtp: Mtp = .{},
    /// Snapshots the radix prompt cache keeps (`--checkpoint-slots`).
    slots: u32 = 8,
    /// Rank whose graph capture fails on purpose, for the tests of the fallback (-1: none). Local to a rank.
    graph_fail: i32 = -1,
    /// The RCCL library to load first. Local to a rank.
    rccl_lib: Path = .{},

    pub const Error = error{ UnknownKey, BadValue, StepNotAligned };

    /// Whether the decode stream tile is on.
    pub fn streamOn(p: *const Policy) bool {
        return p.kernels != .reference and p.stream != .reference;
    }

    /// Whether the prefill GEMM tile (not the reference block tile) takes the products.
    pub fn gemmOn(p: *const Policy) bool {
        return p.kernels != .reference and p.gemm != .reference;
    }

    /// Whether decode.hip's merged launches are on.
    pub fn fused(p: *const Policy) bool {
        return p.kernels != .reference and p.fuse != .reference;
    }

    /// Whether the DeltaNet prefill runs chunked.
    pub fn chunked(p: *const Policy) bool {
        return p.kernels != .reference and p.gdn != .reference;
    }

    /// What the kernel launches choose under this policy.
    pub fn choices(p: *const Policy) Launcher.Choices {
        const matrix: affine.Matrix = switch (p.matrix) {
            .auto => .auto,
            .on => .on,
            .off => .off,
        };
        return .{ .fuse = p.fused(), .wide = p.wideAttention(), .chunked = p.chunked(), .products = .{ .matrix = matrix, .stream = p.streamOn(), .gemm = p.gemmOn() } };
    }

    /// Whether the 64-row prefill attention tile is on.
    pub fn wideAttention(p: *const Policy) bool {
        return p.attention != .f32;
    }

    /// Whether rounds replay captured graphs on a group of `world` ranks.
    pub fn graphsOn(p: *const Policy, world: usize) bool {
        return switch (p.graphs) {
            .on => true,
            .off => false,
            .auto => world == 1,
        };
    }

    /// Applies `key=value,key=value` (`--policy`'s and TF_POLICY's syntax) over this policy, all of it or none.
    pub fn apply(p: *Policy, text: []const u8) Error!void {
        var next = p.*;
        var it = std.mem.tokenizeAny(u8, text, ", ");
        while (it.next()) |pair| {
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse return error.BadValue;
            try next.set(pair[0..eq], pair[eq + 1 ..]);
        }
        if (next.prefill_step == 0 or next.prefill_step % 64 != 0) return error.StepNotAligned;
        p.* = next;
    }

    fn set(p: *Policy, key: []const u8, value: []const u8) Error!void {
        const eql = std.mem.eql;
        if (eql(u8, key, "matrix")) p.matrix = try parseEnum(Choice, value) else if (eql(u8, key, "attention")) p.attention = try parseEnum(Attention, value) else if (eql(u8, key, "kernels")) p.kernels = try parseEnum(Pick, value) else if (eql(u8, key, "graphs")) p.graphs = try parseEnum(Choice, value) else if (eql(u8, key, "stream")) p.stream = try parseEnum(Pick, value) else if (eql(u8, key, "gemm")) p.gemm = try parseEnum(Pick, value) else if (eql(u8, key, "fuse")) p.fuse = try parseEnum(Pick, value) else if (eql(u8, key, "gdn")) p.gdn = try parseEnum(Pick, value) else if (eql(u8, key, "prefill_step")) {
            p.prefill_step = std.fmt.parseInt(u32, value, 10) catch return error.BadValue;
        } else if (eql(u8, key, "graph_fail")) {
            p.graph_fail = std.fmt.parseInt(i32, value, 10) catch return error.BadValue;
        } else if (eql(u8, key, "rccl_lib")) try p.rccl_lib.set(value) else return error.UnknownKey;
    }

    fn parseEnum(comptime E: type, value: []const u8) Error!E {
        return std.meta.stringToEnum(E, value) orelse error.BadValue;
    }

    /// One line for the start-up log and server info; default and local fields are left out of the keys.
    pub fn format(p: Policy, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("matrix={t},attention={t},kernels={t},graphs={t},prefill_step={d}", .{ p.matrix, p.attention, p.kernels, p.graphs, p.prefill_step });
        if (p.stream != .auto) try w.print(",stream={t}", .{p.stream});
        if (p.gemm != .auto) try w.print(",gemm={t}", .{p.gemm});
        if (p.fuse != .auto) try w.print(",fuse={t}", .{p.fuse});
        if (p.gdn != .auto) try w.print(",gdn={t}", .{p.gdn});
        if (p.graph_fail >= 0) try w.print(",graph_fail={d}", .{p.graph_fail});
        if (p.rccl_lib.slice()) |path| try w.print(",rccl_lib={s}", .{path});
        try w.print("; mtp drafts {d} under {d}, {d} snapshots", .{ p.mtp.drafts, p.mtp.confidence, p.slots });
    }

    pub const word_count = 6;

    /// The policy as a fixed set of words, for rank 0 to send and every rank to use the same.
    pub fn words(p: Policy) [word_count]u32 {
        const conf: u64 = @bitCast(p.mtp.confidence);
        const pack = [_]u32{
            @backingInt(p.matrix), @backingInt(p.attention), @backingInt(p.kernels), @backingInt(p.graphs),
            @backingInt(p.stream), @backingInt(p.gemm),      @backingInt(p.fuse),    @backingInt(p.gdn),
        };
        var nibbles: [2]u32 = .{ 0, 0 };
        for (pack, 0..) |v, i| nibbles[i / 8] |= v << @intCast(4 * (i % 8));
        return .{ nibbles[0], nibbles[1] | @as(u32, p.mtp.drafts) << 24, p.prefill_step, @truncate(conf), @intCast(conf >> 32), p.slots };
    }

    /// `local` with every field rank 0 sent; a rank keeps its own graph fault and library path.
    pub fn fromWords(w: [word_count]u32, local: Policy) Policy {
        var p = local;
        var vals: [8]u32 = undefined;
        for (&vals, 0..) |*v, i| v.* = (w[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
        p.matrix = @fromBackingInt(@intCast(vals[0]));
        p.attention = @fromBackingInt(@intCast(vals[1]));
        p.kernels = @fromBackingInt(@intCast(vals[2]));
        p.graphs = @fromBackingInt(@intCast(vals[3]));
        p.stream = @fromBackingInt(@intCast(vals[4]));
        p.gemm = @fromBackingInt(@intCast(vals[5]));
        p.fuse = @fromBackingInt(@intCast(vals[6]));
        p.gdn = @fromBackingInt(@intCast(vals[7]));
        p.mtp = .{ .drafts = @truncate(w[1] >> 24), .confidence = @bitCast(@as(u64, w[3]) | @as(u64, w[4]) << 32) };
        p.prefill_step = w[2];
        p.slots = w[5];
        return p;
    }

    /// Where TF_POLICY is read from, once, at resolution.
    pub const Env = struct {
        /// Read the process's variables after `vars`.
        process: bool = false,
        vars: []const [2][]const u8 = &.{},

        pub const none: Env = .{};
        pub const current: Env = .{ .process = true };

        fn get(e: Env, name: [:0]const u8) ?[]const u8 {
            for (e.vars) |v| if (std.mem.eql(u8, v[0], name)) return v[1];
            if (!e.process) return null;
            return std.mem.span(std.c.getenv(name.ptr) orelse return null);
        }
    };

    /// What resolution found besides the flags (TF_POLICY), for the start-up line.
    pub const Notes = struct {
        buf: [768]u8 = undefined,
        len: usize = 0,

        fn add(n: *Notes, comptime fmt: []const u8, args: anytype) void {
            const out = std.fmt.bufPrint(n.buf[n.len..], fmt, args) catch return;
            n.len += out.len;
        }

        pub fn text(n: *const Notes) []const u8 {
            return n.buf[0..n.len];
        }
    };

    /// The policy: defaults, then `flags` (`--policy`), then TF_POLICY; `notes` says when TF_POLICY acted.
    pub fn resolve(flags: []const u8, env: Env, notes: *Notes) Error!Policy {
        var p: Policy = .{};
        try p.apply(flags);
        if (env.get("TF_POLICY")) |text| {
            notes.add("TF_POLICY={s}; ", .{text});
            try p.apply(text);
        }
        return p;
    }
};

test "apply sets fields and refuses what it does not know" {
    var p: Policy = .{};
    try p.apply("matrix=off, kernels=reference,prefill_step=2048");
    try std.testing.expectEqual(Choice.off, p.matrix);
    try std.testing.expectEqual(Pick.reference, p.kernels);
    try std.testing.expectEqual(@as(u32, 2048), p.prefill_step);
    try std.testing.expectError(error.UnknownKey, p.apply("wmma=on"));
    try std.testing.expectError(error.UnknownKey, p.apply("mtp_drafts=2"));
    try std.testing.expectError(error.BadValue, p.apply("matrix=maybe"));
    try std.testing.expectError(error.StepNotAligned, p.apply("prefill_step=1000"));
}

test "a policy survives its words" {
    var p: Policy = .{ .mtp = .{ .drafts = 2, .confidence = 0.25 }, .slots = 4, .graph_fail = 1 };
    try p.apply("matrix=on,attention=f32,graphs=off,gdn=reference");
    const back = Policy.fromWords(p.words(), .{ .graph_fail = 1 });
    try std.testing.expectEqualDeep(p, back);
}

test "TF_POLICY speaks after the flags, and the reference switches" {
    var notes: Policy.Notes = .{};
    const env: Policy.Env = .{ .vars = &.{.{ "TF_POLICY", "matrix=on,stream=reference,attention=f32,graphs=off" }} };
    const p = try Policy.resolve("matrix=off", env, &notes);
    try std.testing.expectEqual(Choice.on, p.matrix);
    try std.testing.expect(!p.streamOn() and p.gemmOn() and p.fused() and p.chunked() and !p.wideAttention());
    try std.testing.expect(!p.graphsOn(1) and !p.graphsOn(2));
    try std.testing.expect(std.mem.indexOf(u8, notes.text(), "TF_POLICY=") != null);
    var reference: Policy = .{};
    try reference.apply("kernels=reference");
    try std.testing.expect(!reference.streamOn() and !reference.gemmOn() and !reference.fused() and !reference.chunked());
}

test "graphs follow the group unless the policy says" {
    var p: Policy = .{};
    try std.testing.expect(p.graphsOn(1) and !p.graphsOn(2));
    try p.apply("graphs=on");
    try std.testing.expect(p.graphsOn(2));
}
