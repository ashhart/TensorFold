//! Real EXLR8 dense matrices (TF_EXLR8_DIR) for the Metal and CUDA checks: the reference's digests, trellis jobs, FP64.

const std = @import("std");
const ex = @import("exlr8");
const ref = @import("exlr8_ref");
const st = ex.safetensors;
const dense = ex.dense;
const codec = ex.codec;
const Job = ex.job.Job;
const a = std.heap.page_allocator;
const u = 0x1p-24;

pub const files = [_][]const u8{ "exlr8-dense-000", "exlr8-dense-003", "exlr8-dense-040", "exlr8-dense-078", "exlr8-vocab" };

/// The reference loader's sha256 of a matrix's fp32 W (out, in) and of a trellis matrix's codes T (in, out).
pub const Expected = struct { w: ?[]const u8 = null, tiles: ?[]const u8 = null };

/// The fixture's digests for matrix `name` of `file`.
pub fn expected(fixture: std.json.Value, file: []const u8, name: []const u8) !Expected {
    const list = (fixture.object.get("dense") orelse return error.BadFixture).object.get(file) orelse return error.BadFixture;
    for (list.array.items) |m| {
        const o = m.object;
        if (!std.mem.eql(u8, o.get("name").?.string, name)) continue;
        return .{
            .w = if (o.get("w_sha256")) |v| v.string else null,
            .tiles = if (o.get("tiles_sha256")) |v| v.string else null,
        };
    }
    return error.MissingFixtureMatrix;
}

pub fn fixtureJson(gpa: std.mem.Allocator, io: std.Io) !std.json.Parsed(std.json.Value) {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, "zig/tests/fixtures/exlr8/oracle.json", gpa, .limited(16 << 20));
    defer gpa.free(text);
    return std.json.parseFromSlice(std.json.Value, gpa, text, .{ .allocate = .alloc_always });
}

fn f16At(t: st.Tensor, i: usize) f32 {
    return @as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)));
}

/// A trellis matrix as phase-one projection jobs over two output blocks, with every tile run for the codes digest.
pub const Trellis = struct {
    m: dense.Matrix,
    entry_bytes: usize,
    /// Every 64-tile run in T (in, out) block order: input block, then output block.
    runs: []u64,
    bases: []u64,
    jobs: []Job,
    /// Output blocks the jobs cover, in job order.
    outs: [2]usize,
    /// suh's channels (own rotation) or its 128-block magnitudes (shared rotation), then svh.
    halves: []f16,
    signs: []i8,
    shared: bool,
    svh_at: u32,

    pub fn init(m: dense.Matrix) !Trellis {
        const r = m.record;
        const tb = 64 * codec.tileBytes(r.k);
        const in_blocks = r.in / 128;
        const out_blocks = r.out / 128;
        const own: usize = if (r.rotation != 0) in_blocks else r.in;
        var t: Trellis = .{ .m = m, .entry_bytes = m.data[0].dim(1), .runs = try a.alloc(u64, in_blocks * out_blocks), .bases = try a.alloc(u64, 2 * in_blocks), .jobs = try a.alloc(Job, 2), .outs = .{ 0, out_blocks - 1 }, .halves = try a.alloc(f16, own + r.out), .signs = try a.alloc(i8, r.in), .shared = r.rotation != 0, .svh_at = @intCast(own) };
        for (0..in_blocks) |ib| for (0..out_blocks) |ob| {
            t.runs[ib * out_blocks + ob] = t.run(ib, ob, tb);
        };
        // A shared rotation has |suh| constant over each 128-block: its signs rotate once, its magnitudes scale sums.
        for (0..r.in) |i| {
            const v = f16At(m.data[1], i);
            t.signs[i] = if (v < 0) -1 else 1;
            if (!t.shared) {
                t.halves[i] = @floatCast(v);
            } else if (i % 128 == 0) {
                t.halves[i / 128] = @floatCast(@abs(v));
            } else if (@abs(v) != @abs(f16At(m.data[1], i - i % 128))) return error.SuhNotBlockConstant;
        }
        for (0..r.out) |o| t.halves[own + o] = @floatCast(f16At(m.data[2], o));
        for (t.outs, 0..) |ob, ji| {
            for (0..in_blocks) |ib| t.bases[ji * in_blocks + ib] = t.run(ib, ob, tb);
            t.jobs[ji] = .{ .k = r.k, .blocks = @intCast(in_blocks), .base = @intCast(ji * in_blocks), .in_scale = if (t.shared) 0 else ex.job.none, .out_scale = t.svh_at + @as(u32, @intCast(ob * 128)), .y_col = @intCast(ji * 128) };
        }
        return t;
    }

    /// Byte offset of input block ib, output block ob's 64 tiles in M.trellis.
    fn run(t: Trellis, ib: usize, ob: usize, tb: usize) u64 {
        return if (t.m.record.side == .input) ib * t.entry_bytes + ob * tb else ob * t.entry_bytes + ib * tb;
    }

    pub fn deinit(t: Trellis) void {
        a.free(t.runs);
        a.free(t.bases);
        a.free(t.jobs);
        a.free(t.halves);
        a.free(t.signs);
    }

    /// The rotation's arguments for `rows` input rows of `in` fp32 values.
    pub fn rotate(t: Trellis, rows: u32) ex.job.Rotate {
        const n: u32 = @intCast(t.m.record.in);
        return .{ .rows = rows, .x_stride = n, .x_f32 = 1, .r_stride = n, .sign = if (t.shared) 0 else ex.job.none, .scale = if (t.shared) ex.job.none else 0 };
    }

    /// Codes T (in, out) from per-run 128 x 128 blocks in `runs` order.
    pub fn assemble(t: Trellis, blocks: []const u8, out: []u8) void {
        const r = t.m.record;
        const ob_n = r.out / 128;
        for (0..r.in / 128) |ib| for (0..ob_n) |ob| for (0..128) |row| {
            @memcpy(out[(ib * 128 + row) * r.out + ob * 128 ..][0..128], blocks[(ib * ob_n + ob) * 16384 + row * 128 ..][0..128]);
        };
    }

    /// FP64 of rows x (in each) through output blocks `outs`, against `got` (rows x 256 fp32).
    pub fn fp64(t: Trellis, book: *const codec.Codebook, x: []const f32, rows: usize, got: []const f32, check: *ref.Check) !void {
        const r = t.m.record;
        const in_blocks = r.in / 128;
        const codes = try a.alloc(u8, r.in * r.out);
        defer a.free(codes);
        try t.m.trellisCodes(book, codes);
        const job_codes = try a.alloc(u8, r.in * 128);
        defer a.free(job_codes);
        const suh = try a.alloc(f32, r.in);
        defer a.free(suh);
        t.m.suh(suh);
        const xin = try a.alloc(f64, 2 * r.in);
        defer a.free(xin);
        const scales = try a.alloc(f32, in_blocks);
        defer a.free(scales);
        for (scales, 0..) |*s, b| s.* = if (t.shared) @abs(suh[128 * b]) else 1;
        var svh: [128]f32 = undefined;
        // the kernels' roundings: rotation, one FMA a weight in 128-term chains, one block FMA a block, the transform
        const widen = @as(f64, @floatFromInt(in_blocks + 200)) / 256.0;
        for (t.outs, 0..) |ob, ji| {
            for (0..r.in) |k| @memcpy(job_codes[k * 128 ..][0..128], codes[k * r.out + ob * 128 ..][0..128]);
            for (&svh, 0..) |*v, c| v.* = f16At(t.m.data[2], ob * 128 + c);
            const exact = try ref.Exact.init(a, job_codes, in_blocks, suh, &svh);
            defer exact.deinit(a);
            for (0..rows) |row| {
                for (xin[0..r.in], xin[r.in..], x[row * r.in ..][0..r.in], suh) |*v, *va, xv, s| {
                    v.* = xv;
                    va.* = if (t.shared) @abs(@as(f64, xv)) else @abs(@as(f64, xv) * s);
                }
                var y: [128]f64 = undefined;
                const m = exact.row(xin[0..r.in], xin[r.in..], scales, &y);
                for (0..128) |c| try check.add(got[row * 256 + ji * 128 + c], y[c], widen * @abs(svh[c]) / 128 * m);
            }
        }
    }
};

/// Random fp32 rows of n values, the hidden state's typical size.
pub fn inputs(rows: usize, n: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const x = try a.alloc(f32, rows * n);
    for (x) |*v| v.* = (prng.random().float(f32) - 0.5) * 2;
    return x;
}

pub fn hex(d: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(d, .lower);
}
