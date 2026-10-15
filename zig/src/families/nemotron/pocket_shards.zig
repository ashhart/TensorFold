//! The pocket in the shards: every shared expert widened once (config.json told), later writes in place.
const std = @import("std");
const cfg = @import("config.zig");

const group = 64;

/// The writer over `core`, the backend's own (its shard_edit, safetensors and affine4_host).
pub fn Io(comptime core: type) type {
    const st = core.safetensors;
    const se = core.shard_edit;
    const host4 = core.affine4_host;
    return struct {
        /// The learned neurons of MoE layers `layers` into `dir`'s shards (widened first if new), as 4-bit; the bytes.
        pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: cfg.Config, layers: []const usize, neurons: usize, up: []f32, down: []f32) !usize {
            const d = c.hidden;
            const own = c.shared_width;
            if (neurons % group != 0 or d % group != 0 or own % group != 0) return error.BadPocket;
            const has = try se.configInt(gpa, io, dir, cfg.pocket_key);
            const widen = has == null;
            if (has) |n| if (n != neurons) return error.PocketOfAnotherSize;
            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const arena = arena_state.allocator();
            var files: Files = .{ .gpa = gpa, .io = io, .dir = dir, .paths = .init(gpa) };
            defer files.close();
            var edits: std.ArrayList(se.Replacement) = .empty;
            var bytes: usize = 0;
            for (0..c.layers) |l| {
                if (c.kinds[l] != .moe) continue;
                const at = std.mem.indexOfScalar(usize, layers, l);
                const n = neurons * d;
                const u = if (at) |j| up[j * n ..][0..n] else try zeros(arena, n);
                const v = if (at) |j| down[j * n ..][0..n] else try zeros(arena, n);
                var buf: [96]u8 = undefined;
                const base = try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer.shared_experts.", .{l});
                bytes += try expert(arena, &files, &edits, base, own, neurons, d, u, v);
            }
            if (widen) try se.grow(gpa, io, dir, edits.items) else try se.overwrite(gpa, io, dir, edits.items);
            if (widen) {
                const mtp = try std.fs.path.join(arena, &.{ dir, "mtp-4bit.safetensors" });
                if (hasMtp(gpa, io, mtp)) {
                    var only: std.ArrayList(se.Replacement) = .empty;
                    var none: Files = .{ .gpa = gpa, .io = io, .dir = dir, .single = mtp, .paths = .init(gpa) };
                    defer none.close();
                    const z = try zeros(arena, neurons * d);
                    bytes += try expert(arena, &none, &only, "layers.1.mixer.shared_experts.", own, neurons, d, z, z);
                    _ = try se.rewrite(gpa, io, mtp, only.items);
                }
                try se.setConfig(gpa, io, dir, &.{ .{ "moe_shared_expert_intermediate_size", @intCast(own + neurons) }, .{ cfg.pocket_key, @intCast(neurons) } });
            }
            return bytes;
        }

        /// One layer's `neurons` rounded in place as `write` hands them back: up rows, then each output row.
        pub fn round(gpa: std.mem.Allocator, neurons: usize, d: usize, up: []f32, down: []f32) !void {
            if (neurons % group != 0 or d % group != 0) return error.BadPocket;
            const width = @max(d, neurons);
            const w = try gpa.alloc(u32, width / 8);
            defer gpa.free(w);
            const s = try gpa.alloc(u16, width / group);
            defer gpa.free(s);
            const b = try gpa.alloc(u16, width / group);
            defer gpa.free(b);
            for (0..neurons) |k| {
                const x = up[k * d ..][0..d];
                host4.quantize(x, w[0 .. d / 8], s[0 .. d / group], b[0 .. d / group]);
                host4.dequantize(w[0 .. d / 8], s[0 .. d / group], b[0 .. d / group], x);
            }
            const col = try gpa.alloc(f32, neurons);
            defer gpa.free(col);
            for (0..d) |o| {
                for (col, 0..) |*x, k| x.* = down[k * d + o];
                host4.quantize(col, w[0 .. neurons / 8], s[0 .. neurons / group], b[0 .. neurons / group]);
                host4.dequantize(w[0 .. neurons / 8], s[0 .. neurons / group], b[0 .. neurons / group], col);
                for (col, 0..) |x, k| down[k * d + o] = x;
            }
        }

        fn hasMtp(gpa: std.mem.Allocator, io: std.Io, path: []const u8) bool {
            var f = st.File.open(gpa, io, path) catch return false;
            defer f.close(io);
            return f.get("layers.1.mixer.shared_experts.up_proj.weight") != null;
        }

        /// The shard files a model's tensors are read from: the index's, or one file (the MTP head's).
        const Files = struct {
            gpa: std.mem.Allocator,
            io: std.Io,
            dir: []const u8,
            single: ?[]const u8 = null,
            open: std.StringArrayHashMapUnmanaged(st.File) = .empty,
            paths: std.heap.ArenaAllocator,
            map: ?std.json.Parsed(std.json.Value) = null,

            fn get(fs: *Files, name: []const u8) !st.Tensor {
                const shard = if (fs.single) |x| x else try fs.shardOf(name);
                const slot = try fs.open.getOrPut(fs.gpa, shard);
                if (!slot.found_existing) slot.value_ptr.* = st.File.open(fs.gpa, fs.io, shard) catch |e| {
                    _ = fs.open.pop();
                    return e;
                };
                return slot.value_ptr.get(name) orelse error.MissingTensor;
            }

            /// The path of the shard holding `name`, kept until the files close.
            fn shardOf(fs: *Files, name: []const u8) ![]const u8 {
                const a = fs.paths.allocator();
                if (fs.map == null) {
                    const path = try std.fs.path.join(a, &.{ fs.dir, "model.safetensors.index.json" });
                    const text = std.Io.Dir.cwd().readFileAlloc(fs.io, path, a, .limited(1 << 26)) catch |e| if (e == error.FileNotFound) {
                        fs.single = try std.fs.path.join(a, &.{ fs.dir, "model.safetensors" });
                        return fs.single.?;
                    } else return e;
                    fs.map = try std.json.parseFromSlice(std.json.Value, fs.gpa, text, .{ .allocate = .alloc_always });
                }
                const root = fs.map.?.value;
                if (root != .object) return error.BadIndex;
                const weights = root.object.get("weight_map") orelse return error.BadIndex;
                if (weights != .object) return error.BadIndex;
                const file = weights.object.get(name) orelse return error.MissingTensor;
                if (file != .string) return error.BadIndex;
                return std.fs.path.join(a, &.{ fs.dir, file.string });
            }

            fn close(fs: *Files) void {
                for (fs.open.values()) |*f| f.close(fs.io);
                fs.open.deinit(fs.gpa);
                if (fs.map) |*x| x.deinit();
                fs.paths.deinit();
            }
        };

        /// One shared expert's up_proj and down_proj with `neurons` learned after its `own`, into `edits`; the bytes.
        fn expert(arena: std.mem.Allocator, files: *Files, edits: *std.ArrayList(se.Replacement), base: []const u8, own: usize, neurons: usize, d: usize, u: []f32, v: []f32) !usize {
            const wide = own + neurons;
            var name: [128]u8 = undefined;
            var parts: [6]st.Tensor = undefined;
            const leaves = [_][]const u8{ "up_proj.weight", "up_proj.scales", "up_proj.biases", "down_proj.weight", "down_proj.scales", "down_proj.biases" };
            for (&parts, leaves) |*t, leaf| t.* = try files.get(try std.fmt.bufPrint(&name, "{s}{s}", .{ base, leaf }));
            const rows_had = parts[0].dim(0); // up_proj's rows: `own`, or `wide` once widened
            if (rows_had != own and rows_had != wide) return error.UnexpectedTensor;
            // up_proj: the own rows as they are, then a row a learned neuron
            const uw = try arena.alloc(u32, wide * d / 8);
            const us = try arena.alloc(u16, wide * d / group);
            const ub = try arena.alloc(u16, wide * d / group);
            @memcpy(uw[0 .. own * d / 8], words(parts[0].bytes)[0 .. own * d / 8]);
            @memcpy(us[0 .. own * d / group], halves(parts[1].bytes)[0 .. own * d / group]);
            @memcpy(ub[0 .. own * d / group], halves(parts[2].bytes)[0 .. own * d / group]);
            for (0..neurons) |k| {
                const r = own + k;
                const x = u[k * d ..][0..d];
                host4.quantize(x, uw[r * d / 8 ..][0 .. d / 8], us[r * d / group ..][0 .. d / group], ub[r * d / group ..][0 .. d / group]);
                host4.dequantize(uw[r * d / 8 ..][0 .. d / 8], us[r * d / group ..][0 .. d / group], ub[r * d / group ..][0 .. d / group], x);
            }
            // down_proj: each output row's own columns as they are, then a column a learned neuron
            const had = rows_had; // down_proj's columns, as wide as up_proj had rows
            const dw = try arena.alloc(u32, d * wide / 8);
            const ds = try arena.alloc(u16, d * wide / group);
            const db = try arena.alloc(u16, d * wide / group);
            const col = try arena.alloc(f32, neurons);
            for (0..d) |o| {
                @memcpy(dw[o * wide / 8 ..][0 .. own / 8], words(parts[3].bytes)[o * had / 8 ..][0 .. own / 8]);
                @memcpy(ds[o * wide / group ..][0 .. own / group], halves(parts[4].bytes)[o * had / group ..][0 .. own / group]);
                @memcpy(db[o * wide / group ..][0 .. own / group], halves(parts[5].bytes)[o * had / group ..][0 .. own / group]);
                for (col, 0..) |*x, k| x.* = v[k * d + o];
                const w8 = dw[o * wide / 8 ..][own / 8 ..][0 .. neurons / 8];
                const s8 = ds[o * wide / group ..][own / group ..][0 .. neurons / group];
                const b8 = db[o * wide / group ..][own / group ..][0 .. neurons / group];
                host4.quantize(col, w8, s8, b8);
                host4.dequantize(w8, s8, b8, col);
                for (col, 0..) |x, k| v[k * d + o] = x;
            }
            const made = [_]struct { leaf: []const u8, dtype: st.DType, shape: [2]usize, bytes: []const u8 }{
                .{ .leaf = leaves[0], .dtype = .u32, .shape = .{ wide, d / 8 }, .bytes = std.mem.sliceAsBytes(uw) },
                .{ .leaf = leaves[1], .dtype = .bf16, .shape = .{ wide, d / group }, .bytes = std.mem.sliceAsBytes(us) },
                .{ .leaf = leaves[2], .dtype = .bf16, .shape = .{ wide, d / group }, .bytes = std.mem.sliceAsBytes(ub) },
                .{ .leaf = leaves[3], .dtype = .u32, .shape = .{ d, wide / 8 }, .bytes = std.mem.sliceAsBytes(dw) },
                .{ .leaf = leaves[4], .dtype = .bf16, .shape = .{ d, wide / group }, .bytes = std.mem.sliceAsBytes(ds) },
                .{ .leaf = leaves[5], .dtype = .bf16, .shape = .{ d, wide / group }, .bytes = std.mem.sliceAsBytes(db) },
            };
            var bytes: usize = 0;
            for (made) |m| {
                try edits.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}{s}", .{ base, m.leaf }), .dtype = m.dtype, .shape = try arena.dupe(usize, &m.shape), .bytes = m.bytes });
                bytes += m.bytes.len;
            }
            return bytes;
        }
    };
}

fn zeros(arena: std.mem.Allocator, n: usize) ![]f32 {
    const z = try arena.alloc(f32, n);
    @memset(z, 0);
    return z;
}

fn words(bytes: []const u8) []align(1) const u32 {
    return std.mem.bytesAsSlice(u32, bytes);
}

fn halves(bytes: []const u8) []align(1) const u16 {
    return std.mem.bytesAsSlice(u16, bytes);
}
