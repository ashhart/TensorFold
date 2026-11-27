//! Contract suite for `hy4_rows_argmax`: lowest-ID tie-break, odd vocab, batched==solo, and red controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hr = @import("hy4_route_rows_cuda");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

const nan_rule = "any non-finite logit sets float_bits = NaN (0x7fc00000); finite rows pick max logit with lowest id tie-break";
const served = [_]c.Shape{
    .{ .rows = 1, .n = 60416, .k = 1 },
    .{ .rows = 17, .n = 30208, .k = 1 },
    .{ .rows = 8, .n = 4103, .k = 1 },
};

pub const Fault = enum { none, bad_nan, bad_tie };

pub const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    fault: Fault,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    vocab_lo: u32 = 30208,
    logits: std.ArrayList(f32) = .empty,
    dev_in: u64 = 0,
    dev_out: u64 = 0,
    dev_one: u64 = 0,

    pub fn deinit(self: *Case) void {
        self.logits.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const ctl: ?g.Gate = switch (self.fault) {
            .none => null,
            .bad_nan => .nonfinite,
            .bad_tie => .shapes,
        };
        return .{
            .spec = .{
                .name = self.name,
                .accum = .none,
                .depth = zeroDepth,
                .out = .raw,
                .nan = .{ .stated = nan_rule },
                .tiles = .{ .{}, .{ .size = 256, .min = 1 }, .{} },
                .served = &served,
            },
            .ctx = self,
            .vt = &vt,
            .control = ctl,
        };
    }

    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        self.fill = fill;
        try self.logits.resize(a, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x4152);
        const rng = prng.random();
        for (0..shape.rows) |r| {
            const row = self.logits.items[r * shape.n ..][0..shape.n];
            for (row) |*v| v.* = rng.float(f32) * 8.0 - 4.0;
            if (shape.n >= 2) {
                const top: f32 = 12.5 + @as(f32, @floatFromInt(r));
                const lo_idx = r % (shape.n / 2);
                const hi_idx = shape.n - 1 - (r % (shape.n / 2));
                row[lo_idx] = top;
                row[hi_idx] = top;
            }
        }
        if (fill == .nonfinite) {
            self.logits.items[0] = std.math.nan(f32);
            if (shape.rows > 1) self.logits.items[shape.n + shape.n / 2] = std.math.inf(f32);
            if (shape.rows > 2) self.logits.items[2 * shape.n + shape.n - 1] = -std.math.inf(f32);
        }
        self.dev_in = try self.rig.input(std.mem.sliceAsBytes(self.logits.items));
        self.dev_out = try self.rig.output(shape.rows * 8);
        self.dev_one = try self.rig.scratch(8);
        return true;
    }

    fn launchRows(self: *Case, in_ptr: u64, out_ptr: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(in_ptr);
        a.add(@as(i32, @intCast(self.shape.n)));
        a.add(@as(i32, @intCast(self.vocab_lo)));
        a.add(@as(i32, @intCast(rows)));
        a.add(out_ptr);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = hr.threads } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Case) !void {
        try self.launchRows(self.dev_in, self.dev_out, self.shape.rows);
    }

    pub fn count(self: *const Case) usize {
        return self.shape.rows * 8;
    }

    pub fn reference(_: *Case, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Case, gate: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const raw = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(raw);
        const words: []align(1) const u32 = std.mem.bytesAsSlice(u32, raw);
        const n = self.shape.n;
        for (0..self.shape.rows) |r| {
            const row = self.logits.items[r * n ..][0..n];
            var bad = false;
            var best_v: f32 = -std.math.inf(f32);
            var best_id: usize = std.math.maxInt(usize);
            for (row, 0..) |v, i| {
                if (!std.math.isFinite(v)) {
                    bad = true;
                } else if (v > best_v or (v == best_v and i < best_id)) {
                    best_v = v;
                    best_id = i;
                }
            }
            const got_bits = words[2 * r];
            const got_id = words[2 * r + 1];
            if (bad) {
                if (!std.math.isNan(@as(f32, @bitCast(got_bits)))) return "non-finite logit row did not produce NaN float_bits";
                continue;
            }
            if (got_bits != @as(u32, @bitCast(best_v))) return "argmax max value bits differ from host reference";
            if (got_id != self.vocab_lo + @as(u32, @intCast(best_id))) return "argmax winning id or tie-break differs from lowest-id rule";
            if (gate == .shapes) {
                try self.launchRows(self.dev_in + r * n * 4, self.dev_one, 1);
                if (!std.mem.eql(u8, raw[r * 8 ..][0..8], try self.rig.bytes(self.dev_one, 8)))
                    return "batched argmax row differs from single-row launch";
            }
        }
        if (!try self.rig.intact()) return "hy4_rows_argmax overwrote a guard region";
        return null;
    }
};
