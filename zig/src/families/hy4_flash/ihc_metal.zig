//! Native Metal launches for Hy4-flash's independent hyper-connections.
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ihc = @import("ihc.zig");

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
/// `partials` (72 floats a row) selects the two-launch collapse that spreads a row over eight threadgroups.
pub const Collapse = struct { hidden: Ref, projection: Ref, bias: Ref, scale: Ref, output: Ref, post: ?Ref = null, partials: ?Ref = null };
/// The seam partials a row: 19 sums (8 gate logits, the sum of squares, 10 stream products) x 16 slots.
pub const partial_floats = 19 * 16;
pub const Add = struct { residual: Ref, branch: Ref, post: Ref, output: Ref };
pub const Kernels = struct {
    /// Gate counts fixed at compile time (head: 4, blocks: 8); bit-equal to the generic tf_hy4_ihc_collapse.
    pre4: mtl.Pipeline,
    pre8: mtl.Pipeline,
    /// The same collapse split in two: eight one-simdgroup threadgroups a row, then the gates and the row.
    part: [2]mtl.Pipeline,
    finish: [2]mtl.Pipeline,
    /// Sub-block seams: tail (output norm, residual mix, next collapse's partials), head (gates, row, input norm).
    tail: [2]mtl.Pipeline,
    head: [2]mtl.Pipeline,
    /// tail and the next head in one launch with an in-kernel barrier across a row's 12 threadgroups (decode).
    seam: [2]mtl.Pipeline,
    mix: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Kernels {
        const library = try mtl.Library.fromSource(device, sources.hy4_flash_ihc, mtl.CompileOptions.mlx());
        defer library.deinit();
        var pre4 = try mtl.Pipeline.init(device, library, "tf_hy4_ihc_collapse_g4", false);
        errdefer pre4.deinit();
        var pre8 = try mtl.Pipeline.init(device, library, "tf_hy4_ihc_collapse_g8", false);
        errdefer pre8.deinit();
        var sets: [5][2]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (0..made) |m| sets[m / 2][m % 2].deinit();
        inline for (.{ "tf_hy4_ihc_partials_g", "tf_hy4_ihc_finish_g", "tf_hy4_tail_g", "tf_hy4_head_g", "tf_hy4_seam_g" }, 0..) |name, si| {
            inline for (.{ "4", "8" }, 0..) |g, i| {
                sets[si][i] = try mtl.Pipeline.init(device, library, name ++ g, false);
                made += 1;
            }
        }
        return .{ .pre4 = pre4, .pre8 = pre8, .part = sets[0], .finish = sets[1], .tail = sets[2], .head = sets[3], .seam = sets[4], .mix = try mtl.Pipeline.init(device, library, "tf_hy4_ihc_add", false) };
    }

    pub fn deinit(self: *Kernels) void {
        self.pre4.deinit();
        self.pre8.deinit();
        for (self.part) |p| p.deinit();
        for (self.finish) |p| p.deinit();
        for (self.tail) |p| p.deinit();
        for (self.head) |p| p.deinit();
        for (self.seam) |p| p.deinit();
        self.mix.deinit();
        self.* = undefined;
    }

    pub fn collapse(self: Kernels, e: mtl.ComputeEncoder, b: Collapse, shape: ihc.Shape) !void {
        try shape.check();
        if (shape.gates != ihc.streams and b.post == null) return error.MissingHy4PostGates;
        const which: usize = @intFromBool(shape.gates != ihc.streams);
        if (b.partials) |part| {
            e.setPipeline(self.part[which]);
            for ([_]Ref{ b.hidden, b.projection, part }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
            e.setValue(shape, 3);
            e.dispatchGroups(mtl.Size.of(8, shape.rows, 1), mtl.Size.of(32, 1, 1));
            e.barrier();
            e.setPipeline(self.finish[which]);
            const args = [_]Ref{ b.hidden, part, b.bias, b.scale, b.output, b.post orelse b.hidden };
            for (args, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
            e.setValue(shape, args.len);
            e.dispatchGroups(mtl.Size.of((shape.width + 255) / 256, shape.rows, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        e.setPipeline(if (shape.gates == ihc.streams) self.pre4 else self.pre8);
        const args = [_]Ref{ b.hidden, b.projection, b.bias, b.scale, b.output, b.post orelse b.hidden };
        for (args, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, args.len);
        e.dispatchGroups(mtl.Size.of(shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// The collapse's partial sums only (the first of its two launches).
    pub fn partials(self: Kernels, e: mtl.ComputeEncoder, hidden: Ref, projection: Ref, part: Ref, shape: ihc.Shape) !void {
        try shape.check();
        e.setPipeline(self.part[@intFromBool(shape.gates != ihc.streams)]);
        for ([_]Ref{ hidden, projection, part }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, 3);
        e.dispatchGroups(mtl.Size.of(8, shape.rows, 1), mtl.Size.of(32, 1, 1));
    }

    pub const Head = struct { hidden: Ref, partials: Ref, bias: Ref, scale: Ref, gain: Ref, collapsed: Ref, normalized: Ref, post: Ref };

    /// The collapse's gates and row from its partials, then the input RMS norm: one threadgroup a row.
    pub fn headPass(self: Kernels, e: mtl.ComputeEncoder, b: Head, shape: ihc.Shape, norm_eps: f32) !void {
        try shape.check();
        e.setPipeline(self.head[@intFromBool(shape.gates != ihc.streams)]);
        for ([_]Ref{ b.hidden, b.partials, b.bias, b.scale, b.gain, b.collapsed, b.normalized, b.post }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, 8);
        e.setValue(norm_eps, 9);
        if (shape.width > 12 * 256) return error.BadHy4SeamWidth;
        e.dispatchGroups(mtl.Size.of(12, shape.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub const TailShape = extern struct { rows: u32, width: u32, two: u32, partials: u32, eps: f32, mode: u32 };
    pub const Tail = struct { branch: Ref, branch2: ?Ref = null, gain: Ref, residual: Ref, post: Ref, output: Ref, next: ?Ref = null, partials: Ref };

    /// A sub-block's branch (or two, added) through its output norm, mixed into the streams; with `next` (the next
    /// collapse's projection, `gates` of them) also that collapse's partials.
    pub fn tailPass(self: Kernels, e: mtl.ComputeEncoder, b: Tail, rows: u32, width: u32, eps: f32, gates: u32) void {
        e.setPipeline(self.tail[@intFromBool(gates != ihc.streams)]);
        const args = [_]Ref{ b.branch, b.branch2 orelse b.branch, b.gain, b.residual, b.post, b.output, b.next orelse b.branch, b.partials };
        for (args, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(TailShape{ .rows = rows, .width = width, .two = @intFromBool(b.branch2 != null), .partials = @intFromBool(b.next != null), .eps = eps, .mode = 1 }, 8);
        e.dispatchGroups(mtl.Size.of(12, rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub const Next = struct { bias: Ref, scale: Ref, gain: Ref, collapsed: Ref, normalized: Ref, shape: ihc.Shape, barriers: Ref };

    /// Rows a fused seam launch may hold (all its threadgroups resident for the in-kernel barrier).
    pub const seam_rows = 16;

    /// tailPass then the next collapse's headPass in one launch (rows <= seam_rows, width 3072).
    pub fn seamPass(self: Kernels, e: mtl.ComputeEncoder, b: Tail, n: Next, rows: u32, width: u32, eps: f32) !void {
        if (rows > seam_rows or width != 12 * 256 or b.next == null) return error.BadHy4FusedSeam;
        var next = n.shape;
        next.rows = rows;
        try next.check();
        e.setPipeline(self.seam[@intFromBool(next.gates != ihc.streams)]);
        const args = [_]Ref{ b.branch, b.branch2 orelse b.branch, b.gain, b.residual, b.post, b.output, b.next.?, b.partials };
        for (args, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(TailShape{ .rows = rows, .width = width, .two = @intFromBool(b.branch2 != null), .partials = 1, .eps = eps, .mode = 1 }, 8);
        for ([_]Ref{ n.bias, n.scale, n.gain, n.collapsed, n.normalized }, 9..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(next, 14);
        e.setValue(next.eps, 15);
        e.setBuffer(n.barriers.buffer, n.barriers.offset, 16);
        e.dispatchGroups(mtl.Size.of(12, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// The seam partials of `hidden` alone (the collapse after an Engram layer, or with no tail before it).
    pub fn seamPartials(self: Kernels, e: mtl.ComputeEncoder, hidden: Ref, projection: Ref, part: Ref, shape: ihc.Shape) !void {
        try shape.check();
        e.setPipeline(self.tail[@intFromBool(shape.gates != ihc.streams)]);
        const args = [_]Ref{ hidden, hidden, hidden, hidden, hidden, hidden, projection, part };
        for (args, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(TailShape{ .rows = shape.rows, .width = shape.width, .two = 0, .partials = 1, .eps = shape.eps, .mode = 0 }, 8);
        e.dispatchGroups(mtl.Size.of(12, shape.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn add(self: Kernels, e: mtl.ComputeEncoder, b: Add, shape: ihc.Shape) !void {
        try shape.check();
        if (shape.gates != 2 * ihc.streams) return error.MissingHy4PostGates;
        e.setPipeline(self.mix);
        const args = [_]Ref{ b.residual, b.branch, b.post, b.output };
        for (args, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, args.len);
        e.dispatchThreads(mtl.Size.of(ihc.streams * shape.width, shape.rows, 1), mtl.Size.of(256, 1, 1));
    }
};
