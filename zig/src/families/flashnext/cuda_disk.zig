//! Flash Next's CUDA kept states on disk (--learn-conversations): their image, device copies out and in, the probe.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");
const rows = api.prompt_cache.rows;
const sh = snap.shape;
const Allocator = std.mem.Allocator;

const words = 32; // the n-gram history, the position, whether the MTP head's rows follow

/// The image's parts at `at`: words, then the state's own.
pub fn partsOf(at: u32, head: bool, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    try out.append(a, .{ .bytes = words });
    try body(at, head, out, a);
}

/// DeltaNet tails and states, each attention piece by layer, the tail, the head's.
fn body(at: u32, head: bool, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    try out.append(a, .{ .bytes = sh.linear * sh.conv });
    try out.append(a, .{ .bytes = sh.linear * sh.state });
    try attention(at, sh.attn, out, a);
    try out.append(a, .{ .bytes = sh.tail });
    if (head) try attention(at, 1, out, a);
}

/// As cuda_snap.zig's move lays them: every layer's keys, then values, indexer keys, and pooled rows of `ratio`.
fn attention(at: u32, layers: usize, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    for (0..2 * layers) |_| try out.append(a, .{ .bytes = @as(u64, at) * sh.kv_row, .row = sh.kv_row });
    for (0..layers) |_| try out.append(a, .{ .bytes = @as(u64, at) * sh.index_row, .row = sh.index_row });
    for (0..layers) |_| try out.append(a, .{ .bytes = @as(u64, at / sh.ratio) * sh.index_row, .row = sh.index_row, .per = sh.ratio });
}

/// The disk tier's view of a Flash Next namespace `F` (native/cuda_kept.zig; its Cuda is the prompt cache's target).
pub fn View(comptime F: type) type {
    return struct {
        pub const Target = F.Cuda;
        pub const State = snap.Snap;
        pub const Words = extern struct { history: [2]i64, at: u64, head: u64 };
        pub const probe_seed: u64 = 0x666e;
        pub const one_rank = true; // the states are one tensor rank's

        pub fn driver(c: *Target) *const cuda.Driver {
            return c.engine.driver;
        }

        pub fn stream(c: *Target) cuda.Stream {
            return c.stream;
        }

        pub fn buffers(st: *const State) [2]?cuda.DeviceBuffer {
            return .{ st.buf, null };
        }

        pub fn drafts(c: *const Target) bool {
            return c.head != null;
        }

        pub fn wordsOf(st: *const State) Words {
            return .{ .history = st.history, .at = st.at, .head = @intFromBool(st.head) };
        }

        /// A state read back stands where it was kept, with the head's rows as kept: else it restores nowhere.
        pub fn setWords(st: *State, w: Words) !void {
            if (w.at != st.at or w.head != @intFromBool(st.head)) return error.KeptWordsDiffer;
            st.history = w.history;
        }

        pub fn parts(c: *const Target, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
            try body(at, c.head != null, out, a);
        }

        pub fn save(c: *Target, s: *lanes.Stream, at: u32) !*State {
            return @ptrCast(@alignCast(try F.snap_save(c, s, at)));
        }

        pub fn drop(c: *Target, st: *State) void {
            F.snap_drop(c, st);
        }

        /// A device buffer for `at` tokens with the head's rows if this server drafts, filled by `scatter`.
        pub fn blank(c: *Target, at: u32) !*State {
            const head = c.head != null;
            const st = try c.gpa.create(snap.Snap);
            errdefer c.gpa.destroy(st);
            st.* = .{ .buf = try cuda.DeviceBuffer.alloc(c.engine.driver, snap.bytes(at, head, .full)), .at = at, .history = .{ 0, 0 }, .head = head };
            return st;
        }

        /// A state from a stream that did not draft lacks the head's rows this server's parts name: it stays off disk.
        pub fn whole(c: *const Target, st: *const State) bool {
            return st.head == (c.head != null) and st.rank_id == 0 and st.heads.eql(.full);
        }

        /// What else the states depend on: whether the MTP head's rows are part of them.
        pub fn facts(c: *const Target) [1]u64 {
            return .{@intFromBool(c.head != null)};
        }

        /// The native key extends with policyOf under this domain (native/cuda_disk.zig keptKey).
        pub const policy_domain = "flash-cuda-policy-v1";
        pub fn policy(target: anytype) [13]u64 {
            return policyOf(@as(*const Target, @ptrCast(@alignCast(target))));
        }
    };
}

/// What a prompt pass that writes a kept state was resolved with at load: its rows, kernel forms and expert grids.
pub fn policyOf(c: anytype) [13]u64 {
    const e = c.engine;
    const k = &e.batch_k;
    const f = k.forms;
    var grids = std.hash.Wyhash.init(0); // resident blocks a grid takes, which pick the experts' decode forms
    grids.update(std.mem.asBytes(&k.nvfp4.resident));
    grids.update(std.mem.asBytes(&k.fp8.resident));
    if (k.expert_tile) |x| grids.update(std.mem.asBytes(&x.resident));
    grids.update(std.mem.asBytes(&k.native_lane.prompt_sms)); // a BF16 prompt tile runs over half these SMs' worth
    return .{
        // TF_FLASHNEXT_CHUNK_SOLO and _CHUNK as resolved: a pass's rows alone, and beside a decoding stream
        e.chunk,
        e.busy_chunk,
        @intFromBool(f.gdn_fused),
        @intFromBool(f.hc_fuse),
        @intFromBool(f.in_pass),
        f.readout_rows,
        f.fact_rows,
        // TF_FLASHNEXT_PROMPT_EXPERT_TILE, _PROMPT_BF16 and TF_NVFP4_DECODE_WARPS as the kernels loaded
        if (k.expert_tile) |x| 1 + @as(u64, @backingInt(x.selection)) else 0,
        @intFromBool(k.native_lane.prompt != null),
        // TF_QMMF_HALF_BELOW: a pass of 16 rows or fewer takes the lane's 32-column BF16 and MXFP8 tile below it
        k.native_lane.half_below,
        k.nvfp4.warps,
        grids.final(),
        // a captured set in use, and the entries TF_FLASHNEXT_OWNED runs on our glue beside it
        if (c.set != null) 1 + @as(u64, c.glue.mixed) else 0,
    };
}

test "the image covers a state's device bytes once, its rows split into 256-token blocks, with and without the head" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |head| for ([_]u32{ 1, 3, 256, 1000, 4137 }) |at| {
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try partsOf(at, head, &parts, a);
        try std.testing.expect(rows.image.valid(parts.items, at, rows.block_tokens));
        try std.testing.expectEqual(words + snap.bytes(at, head, .full), rows.image.imageBytes(parts.items));
    };
}

test "the image's parts are cuda_snap.move's pieces in its order: sizes, offsets and rows, with and without the head" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |head| for ([_]u32{ 0, 1, 3, 4, 255, 256, 1000, 4136, 4137 }) |at| {
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try partsOf(at, head, &parts, a);
        try std.testing.expectEqual(@as(u64, words), parts.items[0].bytes);
        const o = snap.order(at, head, .full);
        var i: usize = 1;
        var off: u64 = words; // the image's offset; move's buffer offset is this less the words
        var moved: u64 = 0;
        for (o.p[0..o.len]) |p| {
            if (p.row == 0) { // the DeltaNet's pieces and the tail: one part of every layer's bytes
                try std.testing.expectEqual(@as(u64, p.layers * p.n), parts.items[i].bytes);
                try std.testing.expectEqual(@as(u32, 0), parts.items[i].row);
                off += parts.items[i].bytes;
                moved += p.layers * p.n;
                i += 1;
                continue;
            }
            for (0..p.layers) |_| { // each layer's rows: keys, values, indexer keys, pooled blocks
                const x = parts.items[i];
                try std.testing.expectEqual(@as(u64, p.n), x.bytes);
                try std.testing.expectEqual(@as(u64, p.row), x.row);
                try std.testing.expectEqual(@as(u64, p.per), x.per);
                try std.testing.expectEqual(off - words, moved);
                off += x.bytes;
                moved += p.n;
                i += 1;
            }
        }
        try std.testing.expectEqual(parts.items.len, i);
        try std.testing.expectEqual(@as(u64, snap.bytes(at, head, .full)), moved);
        try std.testing.expectEqual(rows.image.imageBytes(parts.items), off);
    };
}
