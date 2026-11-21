//! DFloat11-coded weights under a Checkpoint's names: groups found in the headers, each decoded once until a release.

const std = @import("std");
const st = @import("safetensors.zig");
const dfloat11 = @import("dfloat11.zig");
const layout = @import("dfloat11_config.zig");
const Checkpoint = @import("checkpoint.zig").Checkpoint;

/// A family's shapes for the names its loader takes with `get`: the files hold each member's element count alone.
pub const Shapes = struct {
    ctx: *const anyopaque,
    of: *const fn (ctx: *const anyopaque, name: []const u8, buf: *[st.max_rank]usize) ?[]const usize,
};

const stream = ".encoded_exponent";
const parts = [_]struct { []const u8, st.DType }{
    .{ ".luts", .u8 },             .{ stream, .u8 },  .{ ".sign_mantissa", .u8 },
    .{ ".output_positions", .u8 }, .{ ".gaps", .u8 }, .{ ".split_positions", .i64 },
};

const Name = struct { group: u32, member: u32 };
const Group = struct {
    prefix: []const u8,
    members: u32,
    /// The whole group as little-endian BF16, from the first `get` of a member until `drop`.
    decoded: ?[]align(64) u8 = null,
};

pub const Coded = struct {
    arena: std.heap.ArenaAllocator,
    bytes_per_thread: u32,
    threads_per_block: u32,
    shapes: ?Shapes,
    names: std.StringHashMapUnmanaged(Name) = .empty,
    groups: std.ArrayList(Group) = .empty,

    /// Coded `name` in `want`'s shape (the family's without one), alive until the next `release`; null: not coded.
    pub fn get(c: *Coded, ck: *Checkpoint, name: []const u8, want: ?[]const usize) !?st.Tensor {
        const at = c.names.get(name) orelse return null;
        const g = &c.groups.items[at.group];
        var buf: [st.max_rank]usize = undefined;
        const shape = want orelse (if (c.shapes) |s| s.of(s.ctx, name, &buf) else null) orelse {
            refuse("{s} is DFloat11-coded, and its loader names no shape for it", .{name});
            return error.CodedShapeUnknown;
        };
        if (shape.len > st.max_rank) return error.RankTooHigh;
        var stored: [parts.len][]const u8 = undefined;
        var part: [320]u8 = undefined;
        for (parts, &stored) |p, *bytes| {
            const full = std.fmt.bufPrint(&part, "{s}{s}", .{ g.prefix, p[0] }) catch return error.MissingTensor;
            bytes.* = ((try ck.stored(full)) orelse return error.MissingTensor).bytes;
        }
        const n = stored[2].len;
        const range = try dfloat11.member(stored[5], g.members, n, at.member);
        var count: usize = 1;
        for (shape) |d| count = std.math.mul(usize, count, d) catch return error.UnexpectedTensor;
        if (count != range.end - range.begin) {
            refuse("{s}: {d} coded elements, expected {any}", .{ name, range.end - range.begin, shape });
            return error.UnexpectedTensor;
        }
        if (g.decoded == null) {
            const bytes = std.math.mul(usize, 2, n) catch return error.UnexpectedTensor;
            const out = try ck.gpa.alignedAlloc(u8, .@"64", bytes);
            errdefer ck.gpa.free(out);
            dfloat11.decode(.{
                .luts = stored[0],
                .encoded_exponent = stored[1],
                .sign_mantissa = stored[2],
                .output_positions = stored[3],
                .gaps = stored[4],
                .bytes_per_thread = c.bytes_per_thread,
                .threads_per_block = c.threads_per_block,
            }, out) catch |err| {
                refuse("DFloat11 group {s}: {t}", .{ g.prefix, err });
                return err;
            };
            g.decoded = out;
        }
        var t: st.Tensor = .{ .dtype = .bf16, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = g.decoded.?[2 * range.begin .. 2 * range.end] };
        @memcpy(t.shape[0..shape.len], shape);
        return t;
    }

    /// Frees every decoded group: tensors `get` returned are gone, and the next `get` decodes again.
    pub fn drop(c: *Coded, gpa: std.mem.Allocator) void {
        for (c.groups.items) |*g| if (g.decoded) |bytes| {
            gpa.free(bytes);
            g.decoded = null;
        };
    }

    pub fn deinit(c: *Coded, gpa: std.mem.Allocator) void {
        c.drop(gpa);
        c.arena.deinit();
        gpa.destroy(c);
    }
};

/// Has `ck` answer the member weight names of its groups (a `<module>.encoded_exponent` and five more tensors each).
pub fn attach(ck: *Checkpoint, config: layout.Config, shapes: ?Shapes) !void {
    std.debug.assert(ck.coded == null);
    const c = try ck.gpa.create(Coded);
    c.* = .{ .arena = .init(ck.gpa), .bytes_per_thread = config.bytes_per_thread, .threads_per_block = config.threads_per_block, .shapes = shapes };
    errdefer {
        c.arena.deinit();
        ck.gpa.destroy(c);
    }
    const a = c.arena.allocator();
    for (ck.files.items) |*f| for (f.names.keys()) |k| {
        if (!std.mem.endsWith(u8, k, stream)) continue;
        const prefix = k[0 .. k.len - stream.len];
        const pattern = (try config.group(prefix)) orelse {
            refuse("DFloat11 group {s}: no pattern of dfloat11_config describes it", .{prefix});
            return error.CodedGroupUnlisted;
        };
        var split: usize = 0;
        for (parts) |p| {
            const e = entry(ck, try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, p[0] })) orelse {
                refuse("DFloat11 group {s}: no {s}", .{ prefix, p[0][1..] });
                return error.CodedGroupPartial;
            };
            if (e.dtype != p[1]) return error.CodedGroupPartial;
            split = e.end - e.begin;
        }
        const members = @max(1, pattern.members.len);
        if (split != 8 * (members - 1)) {
            refuse("DFloat11 group {s}: {d} members listed, {d} split positions stored", .{ prefix, members, split / 8 });
            return error.CodedMembers;
        }
        const group: u32 = @intCast(c.groups.items.len);
        try c.groups.append(a, .{ .prefix = try a.dupe(u8, prefix), .members = @intCast(members) });
        for (0..members) |j| {
            const name = if (pattern.members.len == 0)
                try std.fmt.allocPrint(a, "{s}.weight", .{prefix})
            else
                try std.fmt.allocPrint(a, "{s}.{s}.weight", .{ prefix, pattern.members[j] });
            // a name stored plain as well, or coded by two groups, has no one answer
            const slot = try c.names.getOrPut(a, name);
            if (slot.found_existing or entry(ck, name) != null) {
                refuse("{s} is DFloat11-coded in {s} and also stored elsewhere", .{ name, prefix });
                return error.CodedNameStored;
            }
            slot.value_ptr.* = .{ .group = group, .member = @intCast(j) };
        }
    };
    ck.coded = c;
}

/// A refusal's line for the log; host tests refuse on purpose, and a logged error fails their run.
fn refuse(comptime fmt: []const u8, args: anytype) void {
    if (!@import("builtin").is_test) std.log.err(fmt, args);
}

fn entry(ck: *const Checkpoint, name: []const u8) ?st.Entry {
    for (ck.files.items) |*f| if (f.names.get(name)) |e| return e;
    return null;
}
