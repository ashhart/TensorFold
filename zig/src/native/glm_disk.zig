//! GLM-5.3-Flash's kept states on disk (--learn-conversations): their whole image, from a resident state's slot too.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const glm = tf.glm;
const rows = api.prompt_cache.rows;
const Snap = glm.snapshot.Snap;
const Backend = glm.backend.Backend;
const Allocator = std.mem.Allocator;

/// The store's view of a kept state: its KDA states, then each MLA layer's rows as a whole state lays them out.
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };

fn back(ptr: *anyopaque) *Backend {
    return @ptrCast(@alignCast(ptr));
}

/// One MLA layer's four parts at `at` tokens: keys, indexer keys, indexer gates, pooled index rows (the last partial).
fn mla(sh: anytype, at: u32) [4]rows.image.Part {
    const n: u64 = at;
    return .{
        .{ .bytes = n * sh.key_row, .row = sh.key_row },
        .{ .bytes = n * sh.i_row, .row = sh.i_row },
        .{ .bytes = n * sh.i_row, .row = sh.i_row },
        .{ .bytes = @as(u64, at / sh.kpool + 1) * sh.i_row, .row = sh.i_row, .per = sh.kpool },
    };
}

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    const sh = glm.snapshot.diskShape(&back(ptr).sl.e.c);
    try out.append(a, .{ .bytes = sh.kda });
    for (0..sh.mla) |_| try out.appendSlice(a, &mla(sh, at));
}

/// Ranges of a state's image: from its buffer, or (resident) the MLA rows from the slot or pool region holding them.
fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const sl = back(ptr).sl;
    const snap: *Snap = @ptrCast(@alignCast(saved));
    try sl.flush();
    const sh = glm.snapshot.diskShape(&sl.e.c);
    var view: glm.state.State = undefined;
    const home = if (snap.home != null or snap.held != null) glm.pool.prefixOf(sl, snap, &view) else null;
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        defer o += len;
        const h = home orelse {
            @memcpy(out[o..][0..len], snap.buf.contents()[@intCast(g.off)..][0..len]);
            continue;
        };
        if (g.off < sh.kda) {
            @memcpy(out[o..][0..len], snap.buf.contents()[@intCast(g.off)..][0..len]);
            continue;
        }
        var off = g.off - sh.kda; // an MLA range: its layer, its part, and where in the part
        const layer_parts = mla(sh, snap.at);
        var layer_bytes: u64 = 0;
        for (layer_parts) |p| layer_bytes += p.bytes;
        const mi: usize = @intCast(off / layer_bytes);
        off -= mi * layer_bytes;
        const c = h.mla[mi];
        for ([4]glm.affine.Ref{ c.keys, c.ik, c.ig, c.pool }, layer_parts) |ref, p| {
            if (off < p.bytes) {
                @memcpy(out[o..][0..len], ref.addr()[@intCast(off)..][0..len]);
                break;
            }
            off -= p.bytes;
        }
    }
}

fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    return @ptrCast(try back(ptr).sl.blank(at));
}

fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const snap: *Snap = @ptrCast(@alignCast(saved));
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        @memcpy(snap.buf.contents()[@intCast(g.off)..][0..len], in[o..][0..len]);
        o += len;
    }
}

/// A resident state whose slot has written below it is gone (stale): it cannot be read whole.
fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
    return back(ptr).sl.usable(@ptrCast(@alignCast(saved)));
}

/// What learned and disk states read under: checkpoint, split, chip, OS build, kernel sources, a probe; its rate.
pub fn identity(io: std.Io, eng: *const glm.engine.Engine, sl: *glm.slots.Slots) !struct { id: u64, rate: f64 } {
    var h = std.hash.Wyhash.init(0x62);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const probe = try sl.probe(); // both Macs of a pair run it in step: its prompt pass exchanges with the peer
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    for ([_]u64{ eng.model_hash, eng.c.tp, eng.chunk_rows, @intFromBool(eng.ep != null), eng.k.source_hash, probe }) |x| h.update(std.mem.asBytes(&x));
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(eng.device.name()));
    return .{ .id = h.final(), .rate = 4143.0 / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}
