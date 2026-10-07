//! Captured Qwen operation fixtures use the native checkpoint views and require identical operation outputs.
const std = @import("std");
const tf = @import("tensorfold");
const mtl = @import("metal");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const Bound = struct { buffer: mtl.Buffer, offset: usize = 0, bytes: usize, owned: bool = true };

pub const Checker = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *tf.qwen35.Model,
    dir: []const u8,
    passed: usize = 0,
    failed: usize = 0,
    weights: usize = 0,

    fn file(self: *Checker, name: []const u8) !Bound {
        const path = try std.fs.path.join(self.gpa, &.{ self.dir, name });
        defer self.gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(self.io, path, self.gpa, .unlimited);
        defer self.gpa.free(bytes);
        const a = try tf.npy.parse(bytes);
        const b = try self.model.device.buffer(@max(16, a.data.len), opts);
        @memcpy(b.contents()[0..a.data.len], a.data);
        return .{ .buffer = b, .bytes = a.data.len };
    }

    fn input(self: *Checker, v: std.json.Value) !Bound {
        if (v.object.get("weight")) |name| {
            const t = try self.model.checkpoint.get(name.string);
            const bytes = (t.buffer.contents() + t.offset)[0..t.bytes];
            var hash: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
            const hex = std.fmt.bytesToHex(hash, .lower);
            if (!std.mem.eql(u8, &hex, v.object.get("sha256").?.string)) return error.WeightHashMismatch;
            self.weights += 1;
            return .{ .buffer = t.buffer, .offset = t.offset, .bytes = t.bytes, .owned = false };
        }
        return self.file(v.object.get("file").?.string);
    }

    fn triple(v: std.json.Value) mtl.Size {
        const a = v.array.items;
        return mtl.Size.of(@intCast(a[0].integer), @intCast(a[1].integer), @intCast(a[2].integer));
    }

    fn run(self: *Checker, op: std.json.ObjectMap) !bool {
        const key = op.get("op").?.string;
        const pipeline = self.model.kernels.byKey(key) orelse return error.UnknownQwenKernel;
        var buffers: [24]Bound = undefined;
        var count: usize = 0;
        defer for (buffers[0..count]) |b| if (b.owned) b.buffer.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = self.model.queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipeline);
        for (op.get("inputs").?.array.items) |v| {
            const b = try self.input(v);
            buffers[count] = b;
            count += 1;
            const slot: usize = @intCast(v.object.get("slot").?.integer);
            if (v.object.get("constant")) |c| {
                if (!c.bool) return error.BadConstant;
                enc.setBytes((b.buffer.contents() + b.offset)[0..b.bytes], slot);
            } else enc.setBuffer(b.buffer, b.offset, slot);
        }
        const first = count;
        for (op.get("outputs").?.array.items) |v| {
            const bytes: usize = @intCast(v.object.get("bytes").?.integer);
            const b = try self.model.device.buffer(@max(16, bytes), opts);
            buffers[count] = .{ .buffer = b, .bytes = bytes };
            count += 1;
            enc.setBuffer(b, 0, @intCast(v.object.get("slot").?.integer));
        }
        enc.dispatchThreads(triple(op.get("grid").?), triple(op.get("threadgroup").?));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("{s}: GPU failure: {s}\n", .{ key, msg });
            return error.GpuFailed;
        }
        var ok = true;
        for (op.get("outputs").?.array.items, buffers[first..count]) |v, b| {
            const expected = try self.file(v.object.get("file").?.string);
            defer expected.buffer.deinit();
            const actual = b.buffer.contents()[0..b.bytes];
            if (expected.bytes != b.bytes or !std.mem.eql(u8, actual, expected.buffer.contents()[0..expected.bytes])) {
                ok = false;
                var different: usize = 0;
                for (actual, expected.buffer.contents()[0..b.bytes]) |a, e| different += @intFromBool(a != e);
                std.debug.print("  {s}: {d}/{d} bytes differ\n", .{ v.object.get("file").?.string, different, b.bytes });
            }
        }
        std.debug.print("{s} {s} ({d:.3} ms)\n", .{ if (ok) "ok" else "FAIL", key, cb.gpuSeconds() * 1e3 });
        return ok;
    }

    pub fn check(self: *Checker) !void {
        const path = try std.fs.path.join(self.gpa, &.{ self.dir, "manifest.json" });
        defer self.gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(self.io, path, self.gpa, .limited(1 << 24));
        defer self.gpa.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, self.gpa, bytes, .{});
        defer parsed.deinit();
        for (parsed.value.object.get("ops").?.array.items) |op| {
            const ok = self.run(op.object) catch |err| {
                std.debug.print("FAIL {s}: {s}\n", .{ op.object.get("op").?.string, @errorName(err) });
                self.failed += 1;
                continue;
            };
            if (ok) self.passed += 1 else self.failed += 1;
        }
        std.debug.print("{d} passed, {d} failed, 0 skipped; {d} checkpoint weights hashed\n", .{ self.passed, self.failed, self.weights });
        if (self.failed != 0) return error.QwenOperationMismatch;
    }
};
