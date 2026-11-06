//! Mixed-TP dumps: the format, its writers and its parser. Version 2 adds phases; version 3 adds canonical bootstrap.
const std = @import("std");
const collective = @import("tp").collective;
const DType = collective.DType;
const Op = collective.Op;

pub const magic = "TFTPDMP1";
pub const version: u32 = 1;
pub const phase_version: u32 = 2;
pub const canonical_version: u32 = 3;
pub const Phase = enum(u8) { legacy, mixer, ffn, bootstrap };

pub const Kind = enum(u32) { collective = 1, logits = 2 };

/// Which collective a record holds; `exchange` and `send` name their peer, `recv` the rank it came from.
pub const Call = enum(u8) { all_reduce, all_gather, exchange };

pub const Error = error{ BadMagic, BadVersion, Truncated, BadKind, BadEnum, OutOfMemory };

/// How tokens were chosen; a teacher-forced run is fed another run's tokens, and its `chosen` stays its own arg-max.
pub const Mode = enum(u8) { greedy, teacher_forced };

pub const Header = struct {
    format_version: u32 = version,
    rank: u32,
    world: u32,
    mode: Mode = .greedy,
    /// For a teacher-forced run, `TokenHash` over the tokens it was fed; 0 otherwise.
    fed_hash: u64 = 0,
    /// Free text naming the box and backend, e.g. "metal m5-max" or "cuda gb10".
    label: []const u8,
    /// What produced the dump, by convention "commit=... model=... prompts=<sha256> clock=<MHz>".
    provenance: []const u8 = "",
};

pub const Collective = struct {
    phase: Phase = .legacy,
    seq: u64,
    layer: u32,
    call: Call,
    dtype: DType,
    op: Op,
    peer: u32,
    send: []const u8,
    recv: []const u8,
};

pub const Logits = struct {
    /// Index of the prompt in the run's prompt file; positions count from 0 within each prompt.
    prompt: u32,
    position: u64,
    chosen: u32,
    /// Raw little-endian f32 bytes, `vocab` of them.
    bytes: []const u8,

    pub fn vocab(l: Logits) usize {
        return l.bytes.len / 4;
    }

    pub fn at(l: Logits, i: usize) f32 {
        return @bitCast(std.mem.readInt(u32, l.bytes[4 * i ..][0..4], .little));
    }
};

// ---------------------------------------------------------------- writing

pub fn writeHeader(w: *std.Io.Writer, h: Header) std.Io.Writer.Error!void {
    try w.writeAll(magic);
    try w.writeInt(u32, h.format_version, .little);
    try w.writeInt(u32, h.rank, .little);
    try w.writeInt(u32, h.world, .little);
    try w.writeAll(&.{ @backingInt(h.mode), 0, 0, 0 });
    try w.writeInt(u64, h.fed_hash, .little);
    try w.writeInt(u32, @intCast(h.label.len), .little);
    try w.writeAll(h.label);
    try w.writeInt(u32, @intCast(h.provenance.len), .little);
    try w.writeAll(h.provenance);
}

pub fn writeCollective(w: *std.Io.Writer, c: Collective) (std.Io.Writer.Error || error{UnsupportedPhaseVersion})!void {
    return writeCollectiveVersion(w, version, c);
}

pub fn writeCollectiveVersion(w: *std.Io.Writer, format_version: u32, c: Collective) (std.Io.Writer.Error || error{UnsupportedPhaseVersion})!void {
    if ((format_version != version and format_version != phase_version and format_version != canonical_version) or
        (format_version == version and c.phase != .legacy) or
        (format_version == phase_version and c.phase == .bootstrap)) return error.UnsupportedPhaseVersion;
    try w.writeInt(u32, @backingInt(Kind.collective), .little);
    try w.writeInt(u64, c.seq, .little);
    try w.writeInt(u32, c.layer, .little);
    try w.writeAll(&.{ @backingInt(c.call), @backingInt(c.dtype), @backingInt(c.op), @backingInt(c.phase) });
    try w.writeInt(u32, c.peer, .little);
    try w.writeInt(u64, c.send.len, .little);
    try w.writeInt(u64, c.recv.len, .little);
    try w.writeAll(c.send);
    try w.writeAll(c.recv);
}

/// `logits` are host f32 values for one position, in vocabulary order.
pub fn writeLogits(w: *std.Io.Writer, prompt: u32, position: u64, chosen: u32, logits: []const f32) std.Io.Writer.Error!void {
    try w.writeInt(u32, @backingInt(Kind.logits), .little);
    try w.writeInt(u32, prompt, .little);
    try w.writeInt(u64, position, .little);
    try w.writeInt(u32, chosen, .little);
    try w.writeInt(u32, @intCast(logits.len), .little);
    for (logits) |v| try w.writeInt(u32, @bitCast(v), .little);
}

/// FNV-1a 64 over (prompt u32, position u64, token u32) in (prompt, position) order; a fed run stores it as `fed_hash`.
pub const TokenHash = struct {
    h: u64 = 0xcbf29ce484222325,

    pub fn add(t: *TokenHash, prompt: u32, position: u64, token: u32) void {
        var b: [16]u8 = undefined;
        std.mem.writeInt(u32, b[0..4], prompt, .little);
        std.mem.writeInt(u64, b[4..12], position, .little);
        std.mem.writeInt(u32, b[12..16], token, .little);
        for (b) |x| {
            t.h ^= x;
            t.h *%= 0x100000001b3;
        }
    }
};

// ---------------------------------------------------------------- reading

/// One rank's dump, parsed in place: the slices point into the dump's bytes.
pub const Dump = struct {
    header: Header,
    collectives: []Collective,
    logits: []Logits,

    pub fn deinit(d: *Dump, a: std.mem.Allocator) void {
        a.free(d.collectives);
        a.free(d.logits);
    }
};

const Cursor = struct {
    b: []const u8,
    at: usize = 0,

    fn take(c: *Cursor, n: usize) Error![]const u8 {
        if (n > c.b.len - c.at) return error.Truncated;
        defer c.at += n;
        return c.b[c.at..][0..n];
    }

    fn int(c: *Cursor, comptime T: type) Error!T {
        return std.mem.readInt(T, (try c.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }
};

fn enumOf(comptime E: type, v: anytype) Error!E {
    return std.enums.fromInt(E, v) orelse error.BadEnum;
}

fn phaseOf(format_version: u32, byte: u8) Error!Phase {
    if (format_version == version and byte != 0) return error.BadEnum;
    if (format_version == phase_version and byte > 2) return error.BadEnum;
    return enumOf(Phase, byte);
}

pub fn parse(a: std.mem.Allocator, bytes: []const u8) Error!Dump {
    var c: Cursor = .{ .b = bytes };
    if (!std.mem.eql(u8, try c.take(magic.len), magic)) return error.BadMagic;
    const format_version = try c.int(u32);
    if (format_version != version and format_version != phase_version and format_version != canonical_version) return error.BadVersion;
    const rank = try c.int(u32);
    const world = try c.int(u32);
    const flags = try c.take(4);
    const mode = try enumOf(Mode, flags[0]);
    const fed_hash = try c.int(u64);
    const label = try c.take(try c.int(u32));
    const provenance = try c.take(try c.int(u32));
    var cs: std.ArrayList(Collective) = .empty;
    errdefer cs.deinit(a);
    var ls: std.ArrayList(Logits) = .empty;
    errdefer ls.deinit(a);
    while (c.at < bytes.len) {
        switch (try enumOf(Kind, try c.int(u32))) {
            .collective => {
                const seq = try c.int(u64);
                const layer = try c.int(u32);
                const e = try c.take(4);
                const peer = try c.int(u32);
                const ns = try c.int(u64);
                const nr = try c.int(u64);
                try cs.append(a, .{
                    .phase = try phaseOf(format_version, e[3]),
                    .seq = seq,
                    .layer = layer,
                    .call = try enumOf(Call, e[0]),
                    .dtype = try enumOf(DType, e[1]),
                    .op = try enumOf(Op, e[2]),
                    .peer = peer,
                    .send = try c.take(ns),
                    .recv = try c.take(nr),
                });
            },
            .logits => {
                const prompt = try c.int(u32);
                const position = try c.int(u64);
                const chosen = try c.int(u32);
                const vocab = try c.int(u32);
                try ls.append(a, .{ .prompt = prompt, .position = position, .chosen = chosen, .bytes = try c.take(@as(usize, vocab) * 4) });
            },
        }
    }
    return .{
        .header = .{ .format_version = format_version, .rank = rank, .world = world, .mode = mode, .fed_hash = fed_hash, .label = label, .provenance = provenance },
        .collectives = try cs.toOwnedSlice(a),
        .logits = try ls.toOwnedSlice(a),
    };
}
