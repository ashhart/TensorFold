//! Guarded fn_qsa_select contracts against an independent sort-based selected-list oracle.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
pub const ref = @import("contract_qsa_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

pub const Select = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    padded: bool,
    l: ref.Layout = .{ .rows = 0, .nb = 0, .pos0 = 0 },
    scores: []f32 = &.{},
    want: []u8 = &.{},
    score_ptr: u64 = 0,
    pos_ptr: u64 = 0,
    ids_ptr: u64 = 0,
    nk_ptr: u64 = 0,
    sparse_ptr: u64 = 0,

    pub fn deinit(self: *Select) void {
        self.rig.gpu.gpa.free(self.scores);
        self.rig.gpu.gpa.free(self.want);
        self.rig.deinit();
    }

    pub fn kernel(self: *Select) g.Kernel {
        return .{ .spec = ref.spec(self.padded), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Select);

    pub fn prepare(self: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const l = ref.layout(shape, self.padded) orelse return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.scores);
        a.free(self.want);
        self.scores = &.{};
        self.want = &.{};
        self.l = l;
        self.scores = try a.alloc(f32, l.rows * l.nb);
        ref.fill(self.scores, l, fill, seed);
        const result = try ref.select(a, self.scores, l);
        defer result.deinit(a);
        self.want = try result.bytes(a);
        self.score_ptr = try self.rig.input(std.mem.sliceAsBytes(self.scores));
        self.pos_ptr = try self.rig.input(std.mem.asBytes(&self.l.pos0));
        self.ids_ptr = try self.rig.output(l.rows * l.idw * 4);
        self.nk_ptr = try self.rig.output(l.rows * 4);
        self.sparse_ptr = try self.rig.output(l.rows * 4);
        return true;
    }

    pub fn run(self: *Select) !void {
        var args: cuda.Args = .{};
        args.add(self.score_ptr);
        args.add(self.pos_ptr);
        args.add(self.ids_ptr);
        args.add(self.nk_ptr);
        args.add(self.sparse_ptr);
        args.add(@as(c_int, @intCast(self.l.nb)));
        args.add(@as(c_int, @intCast(self.l.idw)));
        args.add(@as(c_int, @intCast(self.l.top)));
        args.add(@as(c_int, @intCast(self.l.ratio)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.l.rows) }, .block = .{ .x = 1024 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Select) usize {
        return self.want.len;
    }

    pub fn reference(self: *Select, want: []f64, mag: []f64) !void {
        for (want, mag, self.want) |*w, *m, byte| {
            w.* = @floatFromInt(byte);
            m.* = 0;
        }
    }

    pub fn check(self: *Select, at: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "selection changed a buffer guard";
        const inputs = try self.rig.inputBytes();
        const score_bytes = std.mem.sliceAsBytes(self.scores);
        if (inputs.len != score_bytes.len + 4 or !std.mem.eql(u8, score_bytes, inputs[0..score_bytes.len]) or
            !std.mem.eql(u8, std.mem.asBytes(&self.l.pos0), inputs[score_bytes.len..]))
            return "selection changed scores or the position input";
        if (at == .nonfinite) {
            for (ref.special_bits) |bits| {
                var found = false;
                for (self.scores) |value| if (@as(u32, @bitCast(value)) == bits) {
                    found = true;
                    break;
                };
                if (!found) return "nonfinite fixture lacks a required signed NaN, infinity or zero";
            }
        }
        return ref.compare(self.want, try self.rig.outputBytes());
    }
};
