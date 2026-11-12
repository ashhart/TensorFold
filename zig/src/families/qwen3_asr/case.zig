//! A reference case from tools/zig/qwen3_asr_case.py: manifest.json and one float32 or int32 file a stage.
const std = @import("std");
const Attention = @import("chunks.zig").Attention;

pub const Stage = struct { shape: []const usize, int: bool };

pub const Case = struct {
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    io: std.Io,
    stages: std.StringArrayHashMapUnmanaged(Stage) = .empty,
    frames: usize = 0,
    text: []const u8 = "",
    attention: Attention = .windows, // the tool's default; --global-attention cases say "global"

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Case {
        var c: Case = .{ .arena = .init(gpa), .dir = undefined, .io = io };
        errdefer c.arena.deinit();
        const a = c.arena.allocator();
        c.dir = try a.dupe(u8, dir);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(1 << 22));
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (doc != .object) return error.BadCase;
        const st = doc.object.get("stages") orelse return error.BadCase;
        if (st != .object) return error.BadCase;
        var it = st.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .object) return error.BadCase;
            const shape = e.value_ptr.object.get("shape") orelse return error.BadCase;
            const kind = e.value_ptr.object.get("dtype") orelse return error.BadCase;
            if (shape != .array or kind != .string) return error.BadCase;
            const dims = try a.alloc(usize, shape.array.items.len);
            for (dims, shape.array.items) |*d, v| d.* = if (v == .integer and v.integer >= 0) @intCast(v.integer) else return error.BadCase;
            try c.stages.put(a, e.key_ptr.*, .{ .shape = dims, .int = std.mem.eql(u8, kind.string, "i32") });
        }
        if (doc.object.get("frames")) |f| c.frames = if (f == .integer and f.integer >= 0) @intCast(f.integer) else return error.BadCase;
        if (doc.object.get("text")) |t| c.text = if (t == .string) t.string else return error.BadCase;
        if (doc.object.get("attention")) |t| c.attention = if (t == .string) std.meta.stringToEnum(Attention, t.string) orelse return error.BadCase else return error.BadCase;
        return c;
    }

    pub fn deinit(c: *Case) void {
        c.arena.deinit();
    }

    pub fn count(s: Stage) usize {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        return n;
    }

    fn bytes(c: *Case, gpa: std.mem.Allocator, name: []const u8, want_int: bool) !struct { Stage, []u8 } {
        const s = c.stages.get(name) orelse return error.NoSuchStage;
        if (s.int != want_int) return error.StageType;
        var buf: [256]u8 = undefined;
        const path = try std.fs.path.join(gpa, &.{ c.dir, try std.fmt.bufPrint(&buf, "{s}.bin", .{name}) });
        defer gpa.free(path);
        const raw = try std.Io.Dir.cwd().readFileAlloc(c.io, path, gpa, .limited(1 << 32));
        if (raw.len != 4 * count(s)) {
            gpa.free(raw);
            return error.StageSize;
        }
        return .{ s, raw };
    }

    /// Stage `name`'s float32 values (the caller frees them).
    pub fn floats(c: *Case, gpa: std.mem.Allocator, name: []const u8) ![]f32 {
        _, const raw = try c.bytes(gpa, name, false);
        defer gpa.free(raw);
        const out = try gpa.alloc(f32, raw.len / 4);
        for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
        return out;
    }

    /// Stage `name`'s int32 values as token ids.
    pub fn ids(c: *Case, gpa: std.mem.Allocator, name: []const u8) ![]u32 {
        _, const raw = try c.bytes(gpa, name, true);
        defer gpa.free(raw);
        const out = try gpa.alloc(u32, raw.len / 4);
        for (out, 0..) |*o, i| o.* = std.mem.readInt(u32, raw[4 * i ..][0..4], .little);
        return out;
    }
};

/// How close `got` is to `want`: cosine, the worst absolute difference and its place, the norm of `want`.
pub const Match = struct {
    cosine: f64,
    worst: f64,
    at: usize,
    rms_want: f64,
    rms_diff: f64,

    pub fn of(got: []const f64, want: []const f32) Match {
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        var dd: f64 = 0;
        var m: Match = .{ .cosine = 0, .worst = 0, .at = 0, .rms_want = 0, .rms_diff = 0 };
        for (got, want, 0..) |g, w64, i| {
            const w: f64 = w64;
            dot += g * w;
            gg += g * g;
            ww += w * w;
            const d = @abs(g - w);
            dd += d * d;
            if (d > m.worst or std.math.isNan(d)) {
                m.worst = d;
                m.at = i;
            }
        }
        const n: f64 = @floatFromInt(@max(1, got.len));
        m.cosine = if (gg == 0 or ww == 0) @as(f64, if (gg == ww) 1 else 0) else dot / @sqrt(gg * ww);
        m.rms_want = @sqrt(ww / n);
        m.rms_diff = @sqrt(dd / n);
        return m;
    }
};

test "match: identical rows give cosine 1, a sign flip -1" {
    const want = [_]f32{ 1, -2, 3 };
    const m = Match.of(&.{ 1, -2, 3 }, &want);
    try std.testing.expectEqual(@as(f64, 0), m.worst);
    try std.testing.expectApproxEqAbs(@as(f64, 1), m.cosine, 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, -1), Match.of(&.{ -1, 2, -3 }, &want).cosine, 1e-15);
}
