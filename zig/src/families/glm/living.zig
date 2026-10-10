//! Living Weights on GLM-5.3-Flash: one low-rank change after layer 44's shared-expert down projection.
//! y += 10 * (x a^T) b with x the projection's input [rows, 2048] and y its bf16 output [rows, 4096], in Nemotron's
//! arithmetic exactly (train.metal's tf_train_lora_in, tf_train_gate, tf_train_lora_out, f32 a and b): a row's sums
//! never depend on the window's other rows, so decode windows, prompt chunks and drafted rounds agree bit for bit.
//! Committed blocks are always on (gate -inf) and bf16-exact in RAM, so a saved and reloaded folder computes the same
//! bits; the open block (the lesson in progress) opens on a row only if its cosine with the block's first direction
//! reaches the block's tau. None in use: nothing is encoded at all, the stock engine's bits.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const sidecar = @import("../../core/living_sidecar.zig");
const wts = @import("weights.zig");
const Ref = wts.Ref;

pub const layer: usize = 44;
pub const in: usize = sidecar.in;
pub const out: usize = sidecar.out;
pub const block: usize = sidecar.block;
pub const max_rank: usize = sidecar.max_rank;
pub const max_blocks: usize = max_rank / block;
pub const scale: f32 = sidecar.scale;
/// Nemotron's a_norm squared (adapters.zig): a row's cosine with direction a is (x . a) / sqrt(unit |x|^2).
pub const unit: f32 = 0.577 * 0.577;
/// A gate above any cosine: the block stays shut.
pub const shut: f32 = 2;
/// The most rows one apply takes (a prompt chunk's, prompt.zig max_rows).
pub const max_rows: usize = 4096;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

pub const Living = struct {
    gpa: std.mem.Allocator,
    a: mtl.Buffer, // [max_rank, in] f32: every block's input directions
    b: mtl.Buffer, // [max_rank, out] f32: every block's outputs (the open block's are what a learner trains)
    tau: mtl.Buffer, // [max_blocks] f32: each block's gate (-inf: committed, always on)
    xa: mtl.Buffer, // [max_rows, max_rank] f32 scratch: x a^T, closed blocks zeroed
    xn: mtl.Buffer, // [max_rows] f32 scratch: |x|^2
    gates: mtl.Buffer, // [max_rows, max_blocks] f32 scratch: which blocks opened on each row
    lora_in: mtl.Pipeline,
    gate_p: mtl.Pipeline,
    lora_out: mtl.Pipeline,
    rank: usize = 0, // ranks in use: committed ones, then the open block's 16 if a lesson is open
    committed: usize = 0, // always-on ranks (what a save keeps)
    saved: usize = 0, // committed ranks already on disk (loaded shards, or saved): a save appends [saved, committed)
    shards: usize = 0, // shards loaded
    base_sha: [64]u8 = @splat('0'), // the site's original weight's sha256, hex (the folder's, set when known)
    loaded_sha: ?[32]u8 = null, // the loaded sidecar file's own sha256 (part of the engine's model hash)

    /// Room for every block, none in use; the three kernels compiled from train.metal (Nemotron's learner's).
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device) !*Living {
        const l = try gpa.create(Living);
        errdefer gpa.destroy(l);
        l.* = .{ .gpa = gpa, .a = undefined, .b = undefined, .tau = undefined, .xa = undefined, .xn = undefined, .gates = undefined, .lora_in = undefined, .gate_p = undefined, .lora_out = undefined };
        const sizes = [_]usize{ max_rank * in, max_rank * out, max_blocks, max_rows * max_rank, max_rows, max_rows * max_blocks };
        const fields = [_]*mtl.Buffer{ &l.a, &l.b, &l.tau, &l.xa, &l.xn, &l.gates };
        var got: usize = 0;
        errdefer for (fields[0..got]) |f| f.deinit();
        for (fields, sizes) |f, n| {
            f.* = try device.buffer(n * 4, opts);
            @memset(f.slice(f32, n), 0);
            got += 1;
        }
        @memset(l.tau.slice(f32, max_blocks), shut);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.train, mtl.CompileOptions.mlx());
        defer lib.deinit();
        l.lora_in = try mtl.Pipeline.init(device, lib, "tf_train_lora_in", false);
        errdefer l.lora_in.deinit();
        l.gate_p = try mtl.Pipeline.init(device, lib, "tf_train_gate", false);
        errdefer l.gate_p.deinit();
        l.lora_out = try mtl.Pipeline.init(device, lib, "tf_train_lora_out", false);
        return l;
    }

    pub fn deinit(l: *Living) void {
        for ([_]*mtl.Buffer{ &l.a, &l.b, &l.tau, &l.xa, &l.xn, &l.gates }) |f| f.deinit();
        l.lora_in.deinit();
        l.gate_p.deinit();
        l.lora_out.deinit();
        l.gpa.destroy(l);
    }

    // ---- views (shared memory: the host reads and writes them between command buffers) ----

    /// Every block's input directions, [max_rank, in].
    pub fn aRows(l: *const Living) []f32 {
        return l.a.slice(f32, max_rank * in);
    }
    /// Every block's outputs, [max_rank, out].
    pub fn bRows(l: *const Living) []f32 {
        return l.b.slice(f32, max_rank * out);
    }
    /// Block k's gate.
    pub fn gate(l: *const Living, k: usize) *f32 {
        return &l.tau.slice(f32, max_blocks)[k];
    }
    /// Whether a lesson's gated block is open (the last 16 ranks in use).
    pub fn isOpen(l: *const Living) bool {
        return l.rank > l.committed;
    }
    /// The open block's first rank (Nemotron's Sites.first).
    pub fn first(l: *const Living) usize {
        return l.rank - block;
    }
    /// The open block's outputs [block, out] (the bytes a learner's Adam updates: `b` at first() * out * 4).
    pub fn openB(l: *const Living) []f32 {
        return l.bRows()[l.first() * out ..][0 .. block * out];
    }
    /// The open block's directions [block, in].
    pub fn openA(l: *const Living) []f32 {
        return l.aRows()[l.first() * in ..][0 .. block * in];
    }

    // ---- the forward ----

    /// After the site's projection: y [rows, out] (bf16) += scale (x [rows, in] (bf16) a^T) b, open block gated.
    pub fn apply(l: *const Living, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32) void {
        if (l.rank == 0 or rows == 0) return;
        std.debug.assert(rows <= max_rows and l.rank % block == 0 and l.rank <= max_rank);
        const R: u32 = @intCast(l.rank);
        e.setPipeline(l.lora_in);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(l.a, 0, 1);
        e.setBuffer(l.xa, 0, 2);
        e.setValue([4]u32{ rows, @intCast(in), R, 0 }, 3);
        e.setBuffer(l.xn, 0, 4);
        e.dispatchThreads(mtl.Size.of(32 * R, rows, 1), mtl.Size.of(256, 1, 1));
        e.setPipeline(l.gate_p);
        e.setBuffer(l.xa, 0, 0);
        e.setBuffer(l.xn, 0, 1);
        e.setBuffer(l.tau, 0, 2);
        e.setBuffer(l.gates, 0, 3);
        e.setValue([4]u32{ rows, R, R / 16, 0 }, 4);
        e.setValue(unit, 5);
        e.dispatchThreads(mtl.Size.of(R / 16, rows, 1), mtl.Size.of(R / 16, 1, 1));
        e.setPipeline(l.lora_out);
        e.setBuffer(l.xa, 0, 0);
        e.setBuffer(l.b, 0, 1);
        e.setBuffer(y.buf, y.off, 2);
        e.setValue([4]u32{ rows, @intCast(out), R, 0 }, 3);
        e.setValue(scale, 4);
        e.dispatchThreads(mtl.Size.of(out, rows, 1), mtl.Size.of(256, 1, 1));
    }

    // ---- the learner's runtime API (call between command buffers: nothing of the engine's in flight) ----

    /// A new gated block after the committed ones: directions `dirs` [block, in] (taken as given), gate `tau`, b zero.
    pub fn openBlock(l: *Living, dirs: []const f32, tau: f32) !void {
        if (l.isOpen()) return error.LivingBlockOpen;
        if (l.rank + block > max_rank) return error.LearnedChangeFull;
        if (dirs.len != block * in) return error.LivingShape;
        const at = l.rank;
        @memcpy(l.aRows()[at * in ..][0 .. block * in], dirs);
        @memset(l.bRows()[at * out ..][0 .. block * out], 0);
        l.gate(at / block).* = tau;
        l.rank += block;
    }

    /// The open block's gate (a learner's tau = max keep cosine + 0.02, or -inf for a plain change before commit).
    pub fn setOpenTau(l: *Living, tau: f32) void {
        std.debug.assert(l.isOpen());
        l.gate(l.first() / block).* = tau;
    }

    /// The open block made a committed, always-on change: gate -inf, its a and b rounded to bf16 (what a save keeps).
    pub fn commitOpen(l: *Living) void {
        std.debug.assert(l.isOpen());
        sidecar.roundSlice(l.openA());
        sidecar.roundSlice(l.openB());
        l.gate(l.first() / block).* = -std.math.inf(f32);
        l.committed = l.rank;
    }

    /// The open block taken back out (its lesson left nothing): rank back to the committed ones.
    pub fn closeOpen(l: *Living) void {
        if (!l.isOpen()) return;
        const at = l.first();
        @memset(l.aRows()[at * in ..][0 .. block * in], 0);
        @memset(l.bRows()[at * out ..][0 .. block * out], 0);
        l.gate(at / block).* = shut;
        l.rank = l.committed;
    }

    /// Committed ranks appended as given (rounded to bf16), after any committed ones; refused while a block is open.
    pub fn appendCommitted(l: *Living, a: []const f32, b: []const f32, n: usize) !void {
        if (l.isOpen()) return error.LivingBlockOpen;
        if (n == 0 or n % block != 0 or l.rank + n > max_rank) return error.LivingRanks;
        if (a.len < n * in or b.len < n * out) return error.LivingShape;
        const at = l.rank;
        const da = l.aRows()[at * in ..][0 .. n * in];
        const db = l.bRows()[at * out ..][0 .. n * out];
        @memcpy(da, a[0 .. n * in]);
        @memcpy(db, b[0 .. n * out]);
        sidecar.roundSlice(da);
        sidecar.roundSlice(db);
        for (at / block..(at + n) / block) |k| l.gate(k).* = -std.math.inf(f32);
        l.rank += n;
        l.committed = l.rank;
    }

    /// Every block dropped: the stock forward again.
    pub fn clear(l: *Living) void {
        @memset(l.aRows(), 0);
        @memset(l.bRows(), 0);
        @memset(l.tau.slice(f32, max_blocks), shut);
        l.rank = 0;
        l.committed = 0;
        l.saved = 0;
        l.shards = 0;
    }

    // ---- the sidecar ----

    /// The folder's change: every shard listed in living_weights.index.json summed in order (v2), or the lone v1
    /// living_weights.safetensors; LW_TOPICS=a,b keeps only shards of those topics. False: none, nothing changed.
    /// Every shard's base_sha must match the folder's site weight, and its sha256 the index's.
    pub fn loadDir(l: *Living, dir: []const u8) !bool {
        var arena = std.heap.ArenaAllocator.init(l.gpa);
        defer arena.deinit();
        if ((try sidecar.listShards(arena.allocator(), dir)) == null) return false;
        l.base_sha = try sidecar.siteSha(l.gpa, dir);
        const allow: ?[]const u8 = if (std.c.getenv("LW_TOPICS")) |v| std.mem.span(v) else null;
        const got = (sidecar.loadAll(arena.allocator(), dir, &l.base_sha, allow) catch |err| {
            if (err == error.LivingBaseMismatch) std.log.err("glm: {s}: a Living Weights shard was learned on another checkpoint (this folder's site {s})", .{ dir, &l.base_sha });
            return err;
        }) orelse return false;
        l.clear();
        if (got.ranks == 0) return false;
        @memcpy(l.aRows()[0 .. got.ranks * in], got.a);
        @memcpy(l.bRows()[0 .. got.ranks * out], got.b);
        for (0..got.ranks / block) |k| l.gate(k).* = -std.math.inf(f32);
        l.rank = got.ranks;
        l.committed = got.ranks;
        l.saved = got.ranks;
        l.shards = got.shards;
        l.loaded_sha = got.digest;
        return true;
    }

    /// The committed ranks not yet on disk appended as a new shard (living-NNNN.safetensors + the index, atomic);
    /// earlier shards are never touched. `topic` names the shard ("" none). An open block is not kept.
    pub fn saveDir(l: *Living, dir: []const u8, topic: []const u8) !void {
        if (l.committed <= l.saved) return error.NothingLearned;
        l.base_sha = try sidecar.siteSha(l.gpa, dir);
        const n = l.committed - l.saved;
        const name = try sidecar.appendShard(l.gpa, dir, l.aRows()[l.saved * in ..], l.bRows()[l.saved * out ..], n, topic, &l.base_sha);
        defer l.gpa.free(name);
        std.log.info("glm: Living Weights: {d} ranks saved as {s}/{s}", .{ n, dir, name });
        l.saved = l.committed;
        l.shards += 1;
    }
};

/// `ranks` NEW committed rows (a [ranks, in], b [ranks, out], f32, rounded to bf16 in the file) appended to
/// `dir` as a new shard (v2: living-NNNN.safetensors + living_weights.index.json, atomic; old shards untouched).
/// Returns the shard's file name (caller frees). Round your RAM rows with `sidecar.roundSlice` so RAM == reload.
pub fn saveRows(gpa: std.mem.Allocator, dir: []const u8, a: []const f32, b: []const f32, ranks: usize, topic: []const u8) ![]u8 {
    const sha = try sidecar.siteSha(gpa, dir);
    return sidecar.appendShard(gpa, dir, a, b, ranks, topic, &sha);
}

/// `saveRows`, returning the new shard's bytes written (the learner's save).
pub fn spawnShard(gpa: std.mem.Allocator, dir: []const u8, a: []const f32, b: []const f32, n: usize, topic: []const u8) !usize {
    const name = try saveRows(gpa, dir, a, b, n, topic);
    defer gpa.free(name);
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    const bytes = (try sidecar.readFile(gpa, path)) orelse return error.LivingWrite;
    defer gpa.free(bytes);
    return bytes.len;
}

/// True when `dir` holds a sidecar (the engine then creates the change at load).
pub fn present(gpa: std.mem.Allocator, dir: []const u8) bool {
    for ([_][]const u8{ sidecar.index_name, sidecar.file_name }) |f| {
        const path = std.fs.path.joinZ(gpa, &.{ dir, f }) catch return false;
        defer gpa.free(path);
        if (std.c.access(path, std.c.F_OK) == 0) return true;
    }
    return false;
}

/// The host reference of `apply` for one row: y + scale (x a^T) b in f32, NOT rounded to bf16 (the kernels' formula;
/// summation order differs, so compare the GPU's bf16 with one bf16 step at the value's size).
pub fn reference(a: []const f32, b: []const f32, tau: []const f32, rank: usize, x: []const f32, y: []f32) void {
    var xa: [max_rank]f32 = undefined;
    var xn: f32 = 0;
    for (x) |v| xn += v * v;
    for (0..rank) |q| {
        var s: f32 = 0;
        for (x, a[q * in ..][0..in]) |v, w| s += v * w;
        xa[q] = s;
    }
    for (0..rank / block) |k| {
        const n = unit * xn;
        const open = n > 0 and xa[k * block] >= tau[k] * @sqrt(n);
        if (!open) @memset(xa[k * block ..][0..block], 0);
    }
    for (y, 0..) |*yv, j| {
        var s: f32 = 0;
        for (0..rank) |q| s += xa[q] * b[q * out + j];
        yv.* = yv.* + scale * s;
    }
}
