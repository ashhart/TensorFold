//! A native MTP draft chain's picks and next forwards, after the kept rows have been absorbed.
const chance = @import("cuda_mtp_chance.zig");
const held = @import("cuda_engine_shares.zig");
const cuda = @import("cuda");
const std = @import("std");
const cuda_pdl = @import("cuda_pdl.zig");
pub const max_depth = 15;

pub fn eligible(on: bool, world: usize, local: usize, full: bool, sampled: bool, structured: bool, depth: usize) bool {
    return on and world == 2 and local == 0 and !full and !sampled and !structured and depth > 1 and depth <= max_depth;
}

pub fn run(o: anytype, depth: usize, out: []u32, chances: ?[]f64) !void {
    if (o.chained(depth)) {
        errdefer o.abort();
        for (0..depth) |j| {
            const more = j + 1 < depth;
            try o.queue(j, chances != null, more);
            if (more) try o.next(0, true);
        }
        return o.finish(depth, out, chances);
    }
    for (0..depth) |j| {
        const p = try o.pick(j, chances != null, out[j..][0..1]);
        out[j] = p.token;
        if (chances) |c| c[j] = p.chance;
        if (j + 1 < depth) try o.next(p.token, false);
    }
}

pub fn native(e: anytype, f: anytype) Native(@TypeOf(e.*), @TypeOf(f.*)) {
    return .{ .e = e, .f = f };
}

fn Native(comptime E: type, comptime F: type) type {
    return struct {
        e: *E,
        f: *const F,

        pub fn chained(o: *const @This(), depth: usize) bool {
            const e = o.e;
            return eligible(e.settings.mtp_chain, e.share.world, e.local.len, e.c.dsa, e.cur.sampling != null, e.structured, depth) and e.mtp_pick != null and depth * e.share.world * 16 <= e.pairs.len;
        }

        pub fn pick(o: *@This(), j: usize, sums: bool, token: *[1]u32) !chance.Pick {
            const e = o.e;
            if (sums and !e.structured and e.cur.sampling == null and e.local.len == 0)
                return e.greedyChance(e.bm.logits);
            try e.draw(e.bm.logits, 1, e.cur.st.pos + 1 + j, 0, token);
            return .{ .token = token[0] };
        }

        pub fn queue(o: *@This(), j: usize, sums: bool, more: bool) !void {
            const e = o.e;
            const words = e.pairs.ptr + j * e.share.world * 16;
            try @import("cuda_engine_pick.zig").enqueue(e, e.bm.logits, words, sums);
            try e.stream.synchronize();
            try e.checkGathers();
            if (more) {
                var a: cuda.Args = .{};
                a.add(words);
                a.add(@as(u32, @intCast(e.share.world)));
                a.add(e.bm.ids);
                try cuda_pdl.launch(e.mtp_pick.?, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, e.stream, &a);
            }
        }

        pub fn finish(o: *@This(), depth: usize, out: []u32, chances: ?[]f64) !void {
            const e = o.e;
            const world = e.share.world;
            var all: [max_depth * 2][4]u32 = undefined;
            try cuda.memory.download(e.d, e.pairs.ptr, std.mem.asBytes(&all)[0 .. depth * world * 16]);
            for (0..depth) |j| {
                const p = chance.pick(all[j * world ..][0..world], chances != null);
                out[j] = p.token;
                if (chances) |c| c[j] = p.chance;
            }
        }

        pub fn abort(o: *@This()) void {
            cuda.completion.drain(o.e.d, o.e.stream, "a GLM MTP draft chain did not drain");
        }

        pub fn next(o: *@This(), token: u32, device: bool) !void {
            const e = o.e;
            if (device) try o.f.mtpDevice(e.bm.ids, &.{e.bm.fnormed}) else try o.f.mtp(@ptrCast(&[1]u32{token}), &.{e.bm.fnormed});
            try held.setMtpLen(e, e.cur.st.mtp_len + 1);
            e.cur.in_head += 1;
        }
    };
}
