//! The TP gate's receipts: digests of the state a round reads back, the dump lines, and the merged top two.
const std = @import("std");
const q = @import("qwen27");
const mtl = @import("metal");

const Runner = q.decode_round.Runner;
const Model = q.model.Model;
const TopTwo = q.session.TopTwo;

fn hashBuffer(h: anytype, b: mtl.Buffer) void {
    h.update(b.contents()[0..b.length()]);
}

/// Five tapped planes of one round: post-block hidden rows, plane by plane, so a divergence names its layer.
pub fn tapDigests(r: *const Runner) ![5][32]u8 {
    var out: [5][32]u8 = undefined;
    for (0..out.len) |i| {
        const plane = try r.taps.plane(i);
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(plane.buffer.contents()[plane.offset..][0 .. @as(usize, r.taps.rows) * r.taps.width * 2]);
        h.final(&out[i]);
    }
    return out;
}

/// Everything the next round reads back: both KV caches a layer slot, the GDN states, the slots' offsets.
pub fn stateDigest(r: *const Runner) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (r.caches) |cache| {
        hashBuffer(&h, cache.keys.buffer);
        hashBuffer(&h, cache.values.buffer);
    }
    for (r.gdn.buffers) |b| hashBuffer(&h, b);
    h.update(std.mem.sliceAsBytes(r.offsets));
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// The KV pages and recurrent state a round's arithmetic reads back, without the prompt kernels' scratch.
pub fn committedDigest(r: *const Runner) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (r.caches) |cache| {
        hashBuffer(&h, cache.keys.buffer);
        hashBuffer(&h, cache.values.buffer);
    }
    for (0..r.gdn.budget.layers) |layer| for ([_]bool{ true, false }) |recurrent| {
        if (r.gdn.committed(layer, 0, recurrent)) |bytes| h.update(bytes) else |_| h.update("err");
    };
    h.update(std.mem.sliceAsBytes(r.offsets));
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

pub fn dumpRow(tag: []const u8, m: *const Model, r: *const Runner, dump: bool) void {
    if (!dump) return;
    const values = m.frame.get(.logits).buffer.slice(u16, m.config.vocab);
    const span = m.shard.vocabRows();
    const two = q.session.topTwo(values) catch |err| {
        std.debug.print("[dump] {s} rank={d}/{d} vocab={d} span={d}+{d} top2=error.{s}\n", .{ tag, m.shard.rank, m.shard.ranks, m.config.vocab, span.from, span.len, @errorName(err) });
        return;
    };
    std.debug.print("[dump] {s} rank={d}/{d} vocab={d} span={d}+{d} capacity={d} top2 ids={d},{d} logits={d},{d} first8={any}\n", .{
        tag,        m.shard.rank, m.shard.ranks, m.config.vocab, span.from,    span.len, m.frame.capacity,
        two.ids[0], two.ids[1],   two.logits[0], two.logits[1],  values[0..8],
    });
    const digest = std.fmt.bytesToHex(stateDigest(r), .lower);
    const committed = std.fmt.bytesToHex(committedDigest(r), .lower);
    std.debug.print("[state] {s} kv_gdn_offsets={s} committed={s} ready={d} offsets={any} heads={d} kv_heads={d} k_heads={d} v_heads={d} intermediate={d}\n", .{
        tag, &digest, &committed, r.ready, r.offsets, m.config.heads, m.config.kv_heads, m.config.k_heads, m.config.v_heads, m.config.intermediate,
    });
}

pub fn shaLogits(m: *const Model) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(m.frame.get(.logits).buffer.slice(u16, m.config.vocab)));
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}

/// One rank's local top two, shifted to the global vocabulary, for the mismatch report.
pub fn mergeTop2(lo: [2]u32, local: [2]TopTwo) TopTwo {
    var best = TopTwo{ .ids = .{ 0, 0 }, .logits = .{ -std.math.inf(f32), -std.math.inf(f32) } };
    for (local, 0..) |two, block| for (0..2) |k| {
        const value = two.logits[k];
        const id = two.ids[k] + lo[block];
        // the old fields are read into locals first: assigning to `best` builds the literal in place
        const old_id = best.ids[0];
        const old_value = best.logits[0];
        if (value > old_value) {
            best = .{ .ids = .{ id, old_id }, .logits = .{ value, old_value } };
        } else if (value > best.logits[1]) {
            best = .{ .ids = .{ old_id, id }, .logits = .{ old_value, value } };
        }
    };
    return best;
}
