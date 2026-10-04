//! Every captured Python op on its inputs with our own weights (hash-checked): exit 0 only when all bits equal.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const Model = tf.nemotron.Model;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const Bound = struct { buffer: mtl.Buffer, offset: usize, bytes: usize, owned: bool };

const Checker = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    dir: []const u8,
    ops: usize = 0,
    failed: usize = 0,
    skipped: usize = 0,
    weights_checked: usize = 0,
    weights_bad: usize = 0,
    ids_file: ?[]const u8 = null,

    fn load(self: *Checker, file: []const u8) !Bound {
        const path = try std.fmt.allocPrintSentinel(self.gpa, "{s}/{s}", .{ self.dir, file }, 0);
        defer self.gpa.free(path);
        const f = try mtl.MappedFile.open(path);
        defer f.deinit();
        const a = try tf.npy.parse(f.bytes[0..f.size]);
        const b = try self.m.device.buffer(@max(a.data.len, 64), opts);
        @memcpy(b.contents()[0..a.data.len], a.data);
        return .{ .buffer = b, .offset = 0, .bytes = a.data.len, .owned = true };
    }

    /// Our buffer for a captured model tensor name (tiled projections, fp32 Mamba constants, raw tensors).
    fn resolve(self: *Checker, name: []const u8) ?Bound {
        const w = &self.m.weights;
        inline for (.{ ".weight", "._lane_sbt" }, 0..) |suffix, j| {
            if (std.mem.endsWith(u8, name, suffix)) {
                if (w.linears.get(name[0 .. name.len - suffix.len])) |lin| {
                    const buf = if (j == 0) lin.w else lin.sbt;
                    const bytes = if (j == 0) lin.n * lin.k / 2 else lin.k / 64 * lin.n * 4;
                    return .{ .buffer = buf, .offset = 0, .bytes = bytes, .owned = false };
                }
            }
        }
        if (std.mem.startsWith(u8, name, "fused.mamba.")) {
            var it = std.mem.splitScalar(u8, name["fused.mamba.".len..], '.');
            const layer = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
            const field = it.next() orelse return null;
            const mb = switch (w.layers[layer]) {
                .mamba => |x| x,
                else => return null,
            };
            const c = self.m.config;
            const pick: ?struct { mtl.Buffer, usize } = if (std.mem.eql(u8, field, "conv_w")) .{ mb.conv_w, c.conv_kernel * c.convDim() * 4 } else if (std.mem.eql(u8, field, "conv_b")) .{ mb.conv_b, c.convDim() * 4 } else if (std.mem.eql(u8, field, "a_log")) .{ mb.a_log, c.mamba_heads * 4 } else if (std.mem.eql(u8, field, "d_skip")) .{ mb.d_skip, c.mamba_heads * 4 } else if (std.mem.eql(u8, field, "dt_bias")) .{ mb.dt_bias, c.mamba_heads * 4 } else null;
            const p = pick orelse return null;
            return .{ .buffer = p[0], .offset = 0, .bytes = p[1], .owned = false };
        }
        if (std.mem.startsWith(u8, name, "fused.gate_bias.")) {
            var buf: [96]u8 = undefined;
            const real = std.fmt.bufPrint(&buf, "backbone.layers.{s}.mixer.gate.e_score_correction_bias", .{name["fused.gate_bias.".len..]}) catch return null;
            return self.raw(real);
        }
        return self.raw(name);
    }

    fn raw(self: *Checker, name: []const u8) ?Bound {
        const t = self.m.checkpoint.get(name) catch return null;
        return .{ .buffer = t.buffer, .offset = t.offset, .bytes = t.bytes, .owned = false };
    }

    fn sha(b: Bound) [64]u8 {
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash((b.buffer.contents() + b.offset)[0..b.bytes], &h, .{});
        return std.fmt.bytesToHex(h, .lower);
    }

    /// An input's buffer: our weight when the capture names one (hash-checked), else the captured file.
    fn input(self: *Checker, spec: std.json.ObjectMap) !Bound {
        if (spec.get("weight")) |wn| {
            if (self.resolve(wn.string)) |b| {
                self.weights_checked += 1;
                const want = spec.get("sha256").?.string;
                const got = sha(b);
                if (!std.mem.eql(u8, &got, want)) {
                    self.weights_bad += 1;
                    std.debug.print("  weight {s}: our bytes hash differently from the capture\n", .{wn.string});
                }
                return b;
            }
        }
        const file = (spec.get("file") orelse return error.NoInput).string;
        return self.load(file);
    }

    fn run(self: *Checker, op: std.json.ObjectMap) !void {
        const name = op.get("op").?.string;
        const kind = op.get("kind").?.string;
        if (std.mem.eql(u8, kind, "metal_kernel")) return self.kernelOp(op, name);
        if (std.mem.eql(u8, name, "embed_gather")) {
            self.ids_file = op.get("inputs").?.array.items[0].object.get("file").?.string;
            return;
        }
        if (std.mem.eql(u8, name, "embed_dequantize")) return self.embedOp(op);
        if (std.mem.endsWith(u8, name, "input_rms_norm")) return self.rmsOp(op);
        if (std.mem.eql(u8, name, "argmax")) return self.argmaxOp(op);
        self.skipped += 1;
    }

    fn outputs(self: *Checker, op: std.json.ObjectMap, bufs: []Bound) !void {
        _ = self;
        for (op.get("outputs").?.array.items, bufs) |o, *b| {
            const shape = o.object.get("shape").?.array.items;
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d.integer);
            const size: usize = if (std.mem.eql(u8, o.object.get("dtype").?.string, "bfloat16")) 2 else 4;
            b.* = .{ .buffer = undefined, .offset = 0, .bytes = n * size, .owned = true };
        }
    }

    /// Compare outputs with the capture; `defined` limits route_group's tables to their first UCOUNT entries.
    fn compare(self: *Checker, name: []const u8, op: std.json.ObjectMap, got: []const Bound) !bool {
        var ok = true;
        var ucount: ?usize = null;
        const outs = op.get("outputs").?.array.items;
        for (outs, got) |o, g| {
            if (std.mem.eql(u8, o.object.get("name").?.string, "UCOUNT")) {
                ucount = @intCast(@as([*]const i32, @ptrCast(@alignCast(g.buffer.contents())))[0]);
            }
        }
        for (outs, got) |o, g| {
            const oname = o.object.get("name").?.string;
            const want = try self.load(o.object.get("file").?.string);
            defer want.buffer.deinit();
            var bytes = g.bytes;
            if (ucount != null and (std.mem.eql(u8, oname, "UIDS") or std.mem.eql(u8, oname, "START") or std.mem.eql(u8, oname, "COUNT"))) bytes = ucount.? * 4;
            const a = g.buffer.contents()[0..bytes];
            const b = want.buffer.contents()[0..bytes];
            var differ: usize = 0;
            var first: ?usize = null;
            var i: usize = 0;
            while (i < bytes) : (i += 2) {
                if (a[i] != b[i] or a[i + 1] != b[i + 1]) {
                    differ += 1;
                    if (first == null) first = i;
                }
            }
            if (differ != 0) {
                ok = false;
                std.debug.print("  {s} {s}: {d} of {d} 16-bit words differ (first at byte {d})\n", .{ name, oname, differ, bytes / 2, first.? });
            }
        }
        return ok;
    }

    fn finish(self: *Checker, name: []const u8, ok: bool, gpu_ms: f64) void {
        self.ops += 1;
        if (!ok) self.failed += 1;
        std.debug.print("{s} {s} ({d:.3} ms)\n", .{ if (ok) "ok  " else "FAIL", name, gpu_ms });
    }

    fn dispatch(self: *Checker, pipe: mtl.Pipeline, binds: []const ?Bound, consts: []const ?[]const u8, grid: [3]usize, group: [3]usize) !f64 {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = self.m.queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipe);
        for (binds, consts, 0..) |b, c, i| {
            if (b) |x| enc.setBuffer(x.buffer, x.offset, i) else if (c) |bytes| enc.setBytes(bytes, i);
        }
        enc.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(@min(group[0], grid[0]), @min(group[1], grid[1]), @min(group[2], grid[2])));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        return cb.gpuSeconds() * 1e3;
    }

    fn triple(v: std.json.Value) [3]usize {
        const a = v.array.items;
        return .{ @intCast(a[0].integer), @intCast(a[1].integer), @intCast(a[2].integer) };
    }

    fn kernelOp(self: *Checker, op: std.json.ObjectMap, name: []const u8) !void {
        const kernel = op.get("kernel").?.string;
        var prefix: std.ArrayList(u8) = .empty;
        defer prefix.deinit(self.gpa);
        try prefix.appendSlice(self.gpa, "custom_kernel_");
        try prefix.appendSlice(self.gpa, kernel);
        const template = op.get("template").?.array.items;
        if (template.len > 0) {
            try prefix.append(self.gpa, '_');
            for (template) |t| try prefix.print(self.gpa, "_{d}", .{t.array.items[1].integer});
        }
        try prefix.append(self.gpa, '_');
        const pipe = self.m.kernels.byPrefix(prefix.items) orelse {
            std.debug.print("FAIL {s}: no generated kernel {s}\n", .{ name, prefix.items });
            self.ops += 1;
            self.failed += 1;
            return;
        };
        const ins = op.get("inputs").?.array.items;
        const outs = op.get("outputs").?.array.items;
        var binds: [24]?Bound = @splat(null);
        var consts: [24]?[]const u8 = @splat(null);
        var strides: [2][4]i64 = undefined;
        var slot: usize = 0;
        const attention = std.mem.startsWith(u8, kernel, "lane_attention_partial");
        for (ins) |spec| {
            binds[slot] = try self.input(spec.object);
            slot += 1;
            const iname = spec.object.get("name").?.string;
            if (attention and (std.mem.eql(u8, iname, "K") or std.mem.eql(u8, iname, "V"))) {
                // a contiguous capture of the cache view: [1, KVH, L, D]
                const shape = spec.object.get("shape").?.array.items;
                const kvh: i64 = shape[1].integer;
                const l: i64 = shape[2].integer;
                const d: i64 = shape[3].integer;
                const which: usize = if (iname[0] == 'K') 0 else 1;
                strides[which] = .{ kvh * l * d, l * d, d, 1 };
                consts[slot] = std.mem.asBytes(&strides[which]);
                slot += 1;
            }
        }
        var out_bounds: [8]Bound = undefined;
        try self.outputs(op, out_bounds[0..outs.len]);
        for (out_bounds[0..outs.len]) |*o| {
            o.buffer = try self.m.device.buffer(@max(o.bytes, 64), opts);
            @memset(o.buffer.contents()[0..o.bytes], 0xff);
            binds[slot] = o.*;
            slot += 1;
        }
        const ms = try self.dispatch(pipe, binds[0..slot], consts[0..slot], triple(op.get("grid").?), triple(op.get("threadgroup").?));
        const ok = try self.compare(name, op, out_bounds[0..outs.len]);
        for (binds[0..slot]) |b| if (b) |x| if (x.owned) x.buffer.deinit();
        self.finish(name, ok, ms);
    }

    fn embedOp(self: *Checker, op: std.json.ObjectMap) !void {
        const ids = try self.load(self.ids_file orelse return error.NoIds);
        defer ids.buffer.deinit();
        const rows = ids.bytes / 4;
        const d = self.m.config.hidden;
        const e = self.m.weights.embed;
        var out = [1]Bound{.{ .buffer = try self.m.device.buffer(rows * d * 2, opts), .offset = 0, .bytes = rows * d * 2, .owned = true }};
        defer out[0].buffer.deinit();
        const dims: u32 = @intCast(d);
        const binds = [_]?Bound{ ids, .{ .buffer = e[0].buffer, .offset = e[0].offset, .bytes = 0, .owned = false }, .{ .buffer = e[1].buffer, .offset = e[1].offset, .bytes = 0, .owned = false }, .{ .buffer = e[2].buffer, .offset = e[2].offset, .bytes = 0, .owned = false }, out[0], null };
        const consts = [_]?[]const u8{ null, null, null, null, null, std.mem.asBytes(&dims) };
        const ms = try self.dispatch(self.m.kernels.glue("tf_embed_q4").?, &binds, &consts, .{ d / 2, rows, 1 }, .{ 256, 1, 1 });
        self.finish("embed", try self.compare("embed", op, &out), ms);
    }

    fn rmsOp(self: *Checker, op: std.json.ObjectMap) !void {
        const ins = op.get("inputs").?.array.items;
        const h = try self.input(ins[0].object);
        defer h.buffer.deinit();
        const w = try self.input(ins[1].object);
        const d = self.m.config.hidden;
        const rows = h.bytes / (2 * d);
        var out = [1]Bound{.{ .buffer = try self.m.device.buffer(rows * d * 2, opts), .offset = 0, .bytes = rows * d * 2, .owned = true }};
        defer out[0].buffer.deinit();
        const eps: f32 = self.m.config.eps;
        const dims: u32 = @intCast(d);
        const one: u32 = 1;
        const threads = (d / 4 + 31) / 32 * 32;
        const strides = [2]u32{ dims, dims };
        const binds = [_]?Bound{ h, w, out[0], null, null, null, null };
        const consts = [_]?[]const u8{ null, null, null, std.mem.asBytes(&eps), std.mem.asBytes(&dims), std.mem.asBytes(&one), std.mem.asBytes(&strides) };
        const ms = try self.dispatch(self.m.kernels.glue("tf_rms_mlx").?, &binds, &consts, .{ threads * rows, 1, 1 }, .{ threads, 1, 1 });
        if (w.owned) w.buffer.deinit();
        self.finish("input_rms_norm", try self.compare("input_rms_norm", op, &out), ms);
    }

    fn argmaxOp(self: *Checker, op: std.json.ObjectMap) !void {
        const logits = try self.input(op.get("inputs").?.array.items[0].object);
        defer logits.buffer.deinit();
        const vocab = self.m.config.vocab;
        const rows = logits.bytes / (2 * vocab);
        var out = [1]Bound{.{ .buffer = try self.m.device.buffer(64, opts), .offset = 0, .bytes = rows * 4, .owned = true }};
        defer out[0].buffer.deinit();
        const v: u32 = @intCast(vocab);
        const unmapped: u32 = 0;
        const binds = [_]?Bound{ logits, out[0], null, out[0], null };
        const consts = [_]?[]const u8{ null, null, std.mem.asBytes(&v), null, std.mem.asBytes(&unmapped) };
        const ms = try self.dispatch(self.m.kernels.glue("tf_argmax_bf16").?, &binds, &consts, .{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 });
        self.finish("argmax", try self.compare("argmax", op, &out), ms);
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-nemotron-fixtures MODEL_DIR FIXTURE_DIR [r1|r4]\n", .{});
        std.process.exit(2);
    }
    const only: ?[]const u8 = if (args.len > 3) args[3] else null;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try Model.load(gpa, init.io, args[1], false);
    defer m.deinit();
    std.debug.print("loaded in {d:.2} s (kernels compiled in {d:.2} s)\n", .{ m.load_seconds, m.compile_seconds });

    const manifest_path = try std.fmt.allocPrintSentinel(gpa, "{s}/manifest.json", .{args[2]}, 0);
    defer gpa.free(manifest_path);
    const text = try mtl.MappedFile.open(manifest_path);
    defer text.deinit();
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text.bytes[0..text.size], .{});
    defer parsed.deinit();
    var c = Checker{ .gpa = gpa, .m = m, .dir = args[2] };
    for (parsed.value.object.get("ops").?.array.items) |op| {
        const step = op.object.get("step").?.string;
        if (std.mem.eql(u8, step, "mtp")) continue;
        if (only) |o| if (!std.mem.eql(u8, o, step)) continue;
        c.run(op.object) catch |err| {
            std.debug.print("FAIL {s}: {s}\n", .{ op.object.get("op").?.string, @errorName(err) });
            c.ops += 1;
            c.failed += 1;
        };
    }
    std.debug.print("{d} ops checked, {d} failed, {d} skipped; {d} weights hashed, {d} differ\n", .{ c.ops, c.failed, c.skipped, c.weights_checked, c.weights_bad });
    if (c.failed != 0 or c.weights_bad != 0) std.process.exit(1);
}
