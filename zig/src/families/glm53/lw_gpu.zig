//! Living Weights on the full GLM-5.3: the change's GPU-visible memory and its load from the sidecar.
//! a [max_rank, 2048] and b [max_rank, 6144] f32, whole on every rank (4 MB + 12.6 MB); each canonical slice reads
//! its own columns of a (g53_lw_in's c0). `served` ranks (the always-on, committed blocks) are applied by the forward
//! (forward.zig lwApply); a learner's open, gated block is never served. No sidecar and no --slide: no Lw at all,
//! and the engine is the stock engine bit for bit.
const std = @import("std");
const mtl = @import("metal");
const m = @import("model.zig");
const lwh = @import("glm53_lw");
const sc = lwh.sidecar;

extern "c" fn time(t: ?*i64) i64;
const c_time = time;

pub const Lw = struct {
    a: m.Ref,
    b: m.Ref,
    u: m.Ref, // [max_rows, max_rank]: the slice's x a^T
    tau: []f32, // [max_blocks] host: the learner's gates (the forward reads only `served`)
    served: usize = 0, // ranks applied by the forward: the leading always-on blocks
    on: bool = true,
    path: []const u8, // v1: the sidecar file this change loads from and saves to; v2: the shard index (*.json) in its folder
    index_dir: ?[]const u8 = null, // v2: spawned shards living-NNNN.safetensors + living_weights.index.json
    persisted: usize = 0, // v2: ranks already in shards (a save spawns rows [persisted, ranks) only; shards are never rewritten)
    base_sha: [64]u8,
    tag: u64 = 0, // the loaded sidecar's tag (0: none): every rank must agree before serving
    bufs: [3]mtl.Buffer,

    pub fn aSlice(l: *const Lw) []f32 {
        return l.a.buf.slice(f32, sc.max_rank * sc.in);
    }
    pub fn bSlice(l: *const Lw) []f32 {
        return l.b.buf.slice(f32, sc.max_rank * sc.out);
    }

    /// The change's memory, then the sidecar at `path` if it exists (base_sha must match the model folder's site).
    pub fn open(gpa: std.mem.Allocator, device: mtl.Device, model_dir: []const u8, path: []const u8, max_rows: usize) !*Lw {
        const l = try gpa.create(Lw);
        errdefer gpa.destroy(l);
        const sha = try sc.siteSha(gpa, model_dir);
        const ba = try m.hostBuffer(device, sc.max_rank * sc.in * 4);
        const bb = try m.hostBuffer(device, sc.max_rank * sc.out * 4);
        const bu = try m.hostBuffer(device, @max(max_rows, 1) * sc.max_rank * 4);
        l.* = .{ .a = .{ .buf = ba }, .b = .{ .buf = bb }, .u = .{ .buf = bu }, .tau = try gpa.alloc(f32, sc.max_rank / sc.block), .path = try gpa.dupe(u8, path), .base_sha = sha, .bufs = .{ ba, bb, bu } };
        @memset(l.aSlice(), 0);
        @memset(l.bSlice(), 0);
        @memset(l.tau, lwh.lm.shut);
        if (std.mem.endsWith(u8, path, ".json")) return l.openIndex(gpa);
        const bytes = (try sc.readFile(gpa, path)) orelse {
            std.debug.print("glm53: living weights {s}: none yet (the stock forward until a lesson commits)\n", .{path});
            return l;
        };
        defer gpa.free(bytes);
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const v = try sc.decode(arena_state.allocator(), bytes);
        if (!std.mem.eql(u8, v.base_sha, &sha)) {
            std.debug.print("glm53: living weights {s} were learned on another checkpoint (base_sha {s}, this folder {s}) - refusing\n", .{ path, v.base_sha, &sha });
            return error.LivingBaseMismatch;
        }
        @memcpy(l.aSlice()[0 .. v.ranks * sc.in], v.a);
        @memcpy(l.bSlice()[0 .. v.ranks * sc.out], v.b);
        for (0..v.ranks / sc.block) |k| l.tau[k] = -std.math.inf(f32);
        l.served = v.ranks;
        l.tag = sc.tag(bytes);
        std.debug.print("glm53: living weights {s}: {d} committed ranks ({d} lessons) at layers.77 shared down_proj, tag {x}\n", .{ path, v.ranks, v.ranks / sc.block, l.tag });
        return l;
    }

    /// v2: every listed shard (LW_TOPICS = an allow-list of topics; default all), concatenated in index order.
    fn openIndex(l: *Lw, gpa: std.mem.Allocator) !*Lw {
        l.index_dir = std.fs.path.dirname(l.path) orelse ".";
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const topics: ?[]const u8 = if (std.c.getenv("LW_TOPICS")) |v| std.mem.span(v) else null;
        const v = (try sc.loadIndex(arena_state.allocator(), l.index_dir.?, topics)) orelse {
            std.debug.print("glm53: living weights {s}: no index yet (the stock forward until a lesson commits)\n", .{l.path});
            return l;
        };
        if (v.ranks > 0 and !std.mem.eql(u8, v.base_sha, &l.base_sha)) {
            std.debug.print("glm53: living weights {s} were learned on another checkpoint (base_sha {s}) - refusing\n", .{ l.path, v.base_sha });
            return error.LivingBaseMismatch;
        }
        @memcpy(l.aSlice()[0 .. v.ranks * sc.in], v.a);
        @memcpy(l.bSlice()[0 .. v.ranks * sc.out], v.b);
        for (0..v.ranks / sc.block) |k| l.tau[k] = -std.math.inf(f32);
        l.served = v.ranks;
        l.persisted = v.ranks;
        l.tag = (sc.tag(std.mem.sliceAsBytes(v.b)) ^ sc.tag(std.mem.sliceAsBytes(v.a))) | 1;
        std.debug.print("glm53: living weights index {s}: {d} ranks from shards (topics {s}), tag {x}\n", .{ l.path, v.ranks, topics orelse "all", l.tag });
        return l;
    }

    /// The leading always-on blocks become the served ranks (after a learner moved gates).
    pub fn publish(l: *Lw, rank: usize) void {
        var plain: usize = 0;
        while (plain < rank and l.tau[plain / sc.block] == -std.math.inf(f32)) plain += sc.block;
        l.served = plain;
    }

    /// Rows [0, ranks) into the sidecar (atomic): returns the bytes written.
    pub fn save(l: *Lw, gpa: std.mem.Allocator, ranks: usize) !usize {
        if (l.index_dir) |dir| { // v2: spawn a shard with this session's new rows only (an append; nothing rewritten)
            if (ranks <= l.persisted) return 0;
            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const ar = arena_state.allocator();
            const n = ranks - l.persisted;
            const topic: []const u8 = if (std.c.getenv("LW_TOPIC")) |v| std.mem.span(v) else "general";
            const created = try std.fmt.allocPrint(ar, "{d}", .{c_time(null)});
            const name = try sc.spawn(gpa, ar, dir, l.aSlice()[l.persisted * sc.in ..], l.bSlice()[l.persisted * sc.out ..], n, &l.base_sha, topic, created);
            std.debug.print("glm53: living weights: spawned {s} ({d} ranks, topic {s})\n", .{ name, n, topic });
            l.persisted = ranks;
            return n * (sc.in + sc.out) * 2;
        }
        const bytes = try sc.encode(gpa, l.aSlice(), l.bSlice(), ranks, &l.base_sha);
        defer gpa.free(bytes);
        try sc.writeAtomic(gpa, l.path, bytes);
        return bytes.len;
    }
};
