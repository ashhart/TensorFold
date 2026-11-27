//! Exact byte-for-byte comparison of `hy4_rows` KV/RoPE against `hy4_mla_project` plus `-1` sentinel checks.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hr = @import("hy4_route_rows_cuda");
const c = core.contract.spec;
const g = core.contract.gates;

const RopeParams = extern struct {
    frequencies: [32]f32,
    axes: [32]u32,
};

const RopeView = extern struct {
    rows: u32,
    heads: u32,
    input_stride: u32,
    output_stride: u32,
};

const CacheWrite = extern struct {
    rows: u32,
    capacity: u32,
    epsilon: f32,
};

fn mropeAxis(pair: usize) u32 {
    return if (pair < 30 and (pair % 3) != 0) @intCast(pair % 3) else 0;
}

pub const MlaRef = struct {
    mod: cuda.Module,
    rotary_table: cuda.Function,
    cache_keys: cuda.Function,
    rotate: cuda.Function,

    pub fn load(d: *const cuda.Driver) !MlaRef {
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_mla_project);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .rotary_table = try mod.function("tf_hy4_rotary_table"),
            .cache_keys = try mod.function("tf_hy4_cache_keys"),
            .rotate = try mod.function("tf_hy4_rotate"),
        };
    }

    pub fn unload(self: *MlaRef) void {
        self.mod.unload();
        self.* = undefined;
    }

    fn buildTable(self: *const MlaRef, d: *const cuda.Driver, s: cuda.Stream, a: std.mem.Allocator, freqs: []const f32, pos3: []const i32, rows: usize, use_mrope: bool) !cuda.DeviceBuffer {
        const pos_tri = try a.alloc(i32, rows * 3);
        defer a.free(pos_tri);
        for (0..rows) |r| {
            if (use_mrope) {
                pos_tri[r * 3 + 0] = pos3[0 * rows + r];
                pos_tri[r * 3 + 1] = pos3[1 * rows + r];
                pos_tri[r * 3 + 2] = pos3[2 * rows + r];
            } else {
                pos_tri[r * 3 + 0] = pos3[r];
                pos_tri[r * 3 + 1] = pos3[r];
                pos_tri[r * 3 + 2] = pos3[r];
            }
        }
        var d_pos = try cuda.DeviceBuffer.alloc(d, pos_tri.len * 4);
        defer d_pos.free();
        try d_pos.upload(0, std.mem.sliceAsBytes(pos_tri));
        var table = try cuda.DeviceBuffer.alloc(d, rows * 64 * 2);
        errdefer table.free();
        var params: RopeParams = undefined;
        @memcpy(&params.frequencies, freqs[0..32]);
        for (0..32) |i| params.axes[i] = if (use_mrope) mropeAxis(i) else 0;
        var args: cuda.Args = .{};
        inline for (.{ d_pos.ptr, table.ptr, params }) |v| args.add(v);
        try cuda.launch.launch(self.rotary_table, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, s, &args);
        try s.synchronize();
        return table;
    }

    pub fn checkRope(self: *const MlaRef, case: anytype, got: []const u8) !?[]const u8 {
        const a = case.rig.gpu.gpa;
        const d = case.rig.gpu.d;
        const s = case.rig.s;
        const rows = case.shape.rows;
        const heads: usize = 4;
        const xh: usize = 72;
        const xs: usize = heads * xh + 8;
        const use_mrope = case.kind == .q_rope_mrope;
        const pos_src = if (use_mrope) case.h_i32_b.items else case.h_i32_a.items;
        var table = try self.buildTable(d, s, a, case.h_f32_a.items, pos_src, rows, use_mrope);
        defer table.free();
        // Pack x into tight [rows, heads, 64] input for tf_hy4_rotate.
        const packed_in = try a.alloc(u16, rows * heads * 64);
        defer a.free(packed_in);
        for (0..rows) |r| for (0..heads) |h| {
            @memcpy(packed_in[(r * heads + h) * 64 ..][0..64], case.h_u16_a.items[r * xs + h * xh ..][0..64]);
        };
        var d_in = try cuda.DeviceBuffer.alloc(d, packed_in.len * 2);
        defer d_in.free();
        try d_in.upload(0, std.mem.sliceAsBytes(packed_in));
        var d_out = try cuda.DeviceBuffer.alloc(d, got.len);
        defer d_out.free();
        try d_out.fill8(0xa5, s.handle);
        const view: RopeView = .{ .rows = @intCast(rows), .heads = @intCast(heads), .input_stride = 64, .output_stride = 64 };
        var args: cuda.Args = .{};
        inline for (.{ d_in.ptr, table.ptr, d_out.ptr, view }) |v| args.add(v);
        try cuda.launch.launch(self.rotate, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 32 } }, s, &args);
        try s.synchronize();
        const want = try a.alloc(u8, got.len);
        defer a.free(want);
        try d_out.download(0, want);
        if (!std.mem.eql(u8, got, want)) return "hy4_rows q_rope bytes differ from tf_hy4_rotary_table + tf_hy4_rotate";
        return null;
    }

    pub fn checkKvWrite(self: *const MlaRef, case: anytype, got: []const u8) !?[]const u8 {
        const a = case.rig.gpu.gpa;
        const d = case.rig.gpu.d;
        const s = case.rig.s;
        const rows = case.shape.rows;
        const use_mrope = case.kind == .kv_write_ex and rows > 1;
        const pos_src = if (use_mrope) case.h_i32_c.items else if (case.kind == .kv_write_ex) case.h_i32_b.items else case.h_i32_a.items;
        var table = try self.buildTable(d, s, a, case.h_f32_a.items, pos_src, rows, use_mrope);
        defer table.free();
        // Pack qa[:, q_lora..q_lora+576] into tight [rows, 576] for tf_hy4_cache_keys.
        const packed_in = try a.alloc(u16, rows * 576);
        defer a.free(packed_in);
        const slots = try a.alloc(i32, rows);
        defer a.free(slots);
        for (0..rows) |r| {
            @memcpy(packed_in[r * 576 ..][0..576], case.h_u16_a.items[r * case.qa_stride + case.q_lora ..][0..576]);
            const sp: usize = @intCast(case.h_i32_a.items[r]);
            const slot = sp % case.ring;
            slots[r] = @intCast((if (case.use_multi_cache) r * case.ring else 0) + slot);
        }
        var d_in = try cuda.DeviceBuffer.alloc(d, packed_in.len * 2);
        defer d_in.free();
        try d_in.upload(0, std.mem.sliceAsBytes(packed_in));
        var d_slots = try cuda.DeviceBuffer.alloc(d, rows * 4);
        defer d_slots.free();
        try d_slots.upload(0, std.mem.sliceAsBytes(slots));
        var d_out = try cuda.DeviceBuffer.alloc(d, got.len);
        defer d_out.free();
        try d_out.fill8(0, s.handle);
        const cap: u32 = @intCast(got.len / (576 * 2));
        const cw: CacheWrite = .{ .rows = @intCast(rows), .capacity = cap, .epsilon = case.eps };
        var args: cuda.Args = .{};
        inline for (.{ d_in.ptr, case.dev[1], table.ptr, d_slots.ptr, d_out.ptr, cw }) |v| args.add(v);
        try cuda.launch.launch(self.cache_keys, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, s, &args);
        try s.synchronize();
        const want = try a.alloc(u8, got.len);
        defer a.free(want);
        try d_out.download(0, want);
        if (!std.mem.eql(u8, got, want)) return "hy4_rows kv_write bytes differ from tf_hy4_cache_keys";
        // Negative sentinel (-1) check: must leave poisoned cache completely untouched.
        try d_out.fill8(0xa5, s.handle);
        @memset(slots, -1);
        try d_slots.upload(0, std.mem.sliceAsBytes(slots));
        if (case.kind == .kv_write) {
            try case.rows.kvWrite(s, case.dev[0], case.qa_stride, case.q_lora, case.dev[1], case.dev[2], d_slots.ptr, 0, d_out.ptr, case.ring, rows, case.eps);
        } else {
            const pos3: u64 = if (rows > 1) case.dev[5] else 0;
            try case.rows.kvWriteEx(s, case.dev[0], case.qa_stride, case.q_lora, case.dev[1], case.dev[2], d_slots.ptr, case.dev[4], pos3, rows, 0, d_out.ptr, case.ring, rows, case.eps);
        }
        try s.synchronize();
        try d_out.download(0, want);
        for (want) |b| if (b != 0xa5) return "hy4_rows kv_write modified cache on negative sentinel slot";
        return null;
    }
};

pub fn checkFp64(spec: c.Spec, shape: c.Shape, out: []const u8, want: []const f64, mag: []const f64, allow: ?[]const f64, gate: g.Gate) ?[]const u8 {
    if (gate == .nonfinite) {
        var planted: usize = 0;
        for (want, 0..) |w, i| {
            if (c.classOf(w) != .finite) planted += 1;
            if (c.classOf(c.read(spec.out, out, i)) != c.classOf(w)) return "non-finite class differs from float64";
        }
        if (planted == 0 or planted == want.len) return "non-finite test lacked mixed finite/non-finite outputs";
        return null;
    }
    for (want, mag, 0..) |w, m, i| {
        const limit = if (allow) |al| al[i] else c.bound(spec, shape, w, m);
        const got = c.read(spec.out, out, i);
        const q = if (got == w) 0 else @abs(got - w) / limit;
        if (!(q <= 1)) return "output exceeded float64 bound";
    }
    return null;
}

pub fn checkNum(self: anytype, gate: g.Gate) !?[]const u8 {
    if (gate != .shapes and gate != .nonfinite) return null;
    const a = self.rig.gpu.gpa;
    const n = self.count();
    const got = try a.dupe(u8, try self.rig.outputBytes());
    defer a.free(got);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    const allow = try a.alloc(f64, n);
    defer a.free(allow);
    try self.reference(want, mag);
    try self.bounds(want, mag, allow);
    if (checkFp64(self.kernel().spec, self.shape, got, want, mag, allow, gate)) |why| return why;
    if (gate != .shapes) return null;
    return switch (self.kind) {
        .q_rope, .q_rope_mrope => self.mla.checkRope(self, got),
        .kv_write, .kv_write_ex => self.mla.checkKvWrite(self, got),
        else => null,
    };
}

pub fn checkPagedSentinel(self: anytype) !?[]const u8 {
    const a = self.rig.gpu.gpa;
    const neg = try a.alloc(i32, self.shape.rows);
    defer a.free(neg);
    @memset(neg, -1);
    var d_neg = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, neg.len * 4);
    defer d_neg.free();
    try d_neg.upload(0, std.mem.sliceAsBytes(neg));
    const out_view: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.count() };
    try out_view.fill8(0xa5, self.rig.s.handle);
    try self.rows.kvWritePaged(self.rig.s, self.dev[0], self.qa_stride, self.q_lora, self.dev[1], self.dev[2], d_neg.ptr, self.dev[4], self.dev[5], self.shape.rows, self.dev[6], self.dev[7], self.pt_stride, self.dev_out, self.shape.rows, self.eps);
    try self.rig.s.synchronize();
    const after = try self.rig.outputBytes();
    for (after) |b| if (b != 0xa5) return "hy4_rows_kv_write_paged modified cache on negative sentinel slot";
    if (!try self.rig.intact()) return "hy4_rows_kv_write_paged negative sentinel overwrote guard region";
    return null;
}
