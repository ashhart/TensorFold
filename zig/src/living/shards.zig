//! The pocket in the shards, for any family: every shared expert widened once (config.json told), then in place.
const std = @import("std");
const family = @import("family.zig");
const affine = @import("affine.zig");
const Family = family.Family;
const Layout = family.Layout;

/// The writer over `core`, the backend's own (its shard_edit and safetensors).
pub fn Io(comptime core: type) type {
    const st = core.safetensors;
    const se = core.shard_edit;
    return struct {
        /// The learned neurons of `layers` into `dir`'s shards, quantized as `F` stores them, handed back; the bytes.
        pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, comptime F: Family, lay: Layout, layers: []const usize, neurons: usize, inputs: []const []f32, down: []f32) !usize {
            const q = F.quant;
            const d = lay.hidden;
            const own = lay.width;
            if (inputs.len != F.names.inputs.len) return error.BadPocket;
            if (neurons % q.group != 0 or d % q.group != 0 or own % q.group != 0) return error.BadPocket;
            const has = try se.configIntIn(gpa, io, dir, F.keys.scope, F.keys.pocket);
            const widen = has == null;
            if (has) |n| if (n != neurons) return error.PocketOfAnotherSize;
            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const arena = arena_state.allocator();
            var files: Files = .{ .gpa = gpa, .io = io, .dir = dir, .paths = .init(gpa) };
            defer files.close();
            var edits: std.ArrayList(se.Replacement) = .empty;
            var bytes: usize = 0;
            const n = neurons * d;
            var rows: [F.names.inputs.len][]f32 = undefined;
            for (0..lay.layers) |l| {
                if (!lay.widen.isSet(l)) continue;
                const at = std.mem.indexOfScalar(usize, layers, l);
                for (&rows, inputs) |*r, x| r.* = if (at) |j| x[j * n ..][0..n] else try zeros(arena, n);
                const v = if (at) |j| down[j * n ..][0..n] else try zeros(arena, n);
                var buf: [160]u8 = undefined;
                bytes += try expert(F, arena, &files, &edits, try F.expertBase(&buf, l), own, neurons, d, &rows, v);
            }
            const mtp = if (widen) F.names.mtp else null;
            const z = try zeros(arena, n);
            var zs: [F.names.inputs.len][]f32 = @splat(z);
            if (mtp) |m| if (m.file == null and try files.indexed(try std.fmt.allocPrint(arena, "{s}{s}.weight", .{ m.base, F.names.inputs[0] }))) {
                // the MTP head's expert in the index: grown with the model's own, zero neurons
                bytes += try expert(F, arena, &files, &edits, m.base, own, neurons, d, &zs, z);
            };
            if (widen) try se.grow(gpa, io, dir, edits.items) else try se.overwrite(gpa, io, dir, edits.items);
            if (mtp) |m| if (m.file) |name| {
                // in its own file, zero neurons: that file rewritten whole, apart from the index
                const path = try std.fs.path.join(arena, &.{ dir, name });
                if (hasMtp(F, gpa, io, path, m.base)) {
                    var only: std.ArrayList(se.Replacement) = .empty;
                    var single: Files = .{ .gpa = gpa, .io = io, .dir = dir, .single = path, .paths = .init(gpa) };
                    defer single.close();
                    bytes += try expert(F, arena, &single, &only, m.base, own, neurons, d, &zs, z);
                    _ = try se.rewrite(gpa, io, path, only.items);
                }
            };
            if (widen) try se.setConfigIn(gpa, io, dir, F.keys.scope, &.{ .{ F.keys.width, @intCast(own + neurons) }, .{ F.keys.pocket, @intCast(neurons) } });
            return bytes;
        }

        fn hasMtp(comptime F: Family, gpa: std.mem.Allocator, io: std.Io, path: []const u8, base: []const u8) bool {
            var f = st.File.open(gpa, io, path) catch return false;
            defer f.close(io);
            var buf: [192]u8 = undefined;
            const name = std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ base, F.names.inputs[0] }) catch return false;
            return f.get(name) != null;
        }

        /// The shard files a model's tensors are read from: the index's, or one file (an MTP head's own).
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

            /// Whether the folder's index (or its one shard) holds `name`.
            fn indexed(fs: *Files, name: []const u8) !bool {
                _ = fs.get(name) catch |e| return if (e == error.MissingTensor) false else e;
                return true;
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

        /// One shared expert with `neurons` learned after its `own`: a row a neuron per input, a column out; the bytes.
        fn expert(comptime F: Family, arena: std.mem.Allocator, files: *Files, edits: *std.ArrayList(se.Replacement), base: []const u8, own: usize, neurons: usize, d: usize, inputs: []const []f32, v: []f32) !usize {
            const q = F.quant;
            const g = q.group;
            const wide = own + neurons;
            const per_word = 32 / q.bits;
            var name: [192]u8 = undefined;
            var bytes: usize = 0;
            var had: ?usize = null; // the input projections' rows: `own`, or `wide` once widened
            for (F.names.inputs, inputs) |proj, x| {
                var parts: [3]st.Tensor = undefined;
                for (&parts, [_][]const u8{ "weight", "scales", "biases" }) |*t, leaf| t.* = try files.get(try std.fmt.bufPrint(&name, "{s}{s}.{s}", .{ base, proj, leaf }));
                const rows_had = parts[0].dim(0);
                if (rows_had != own and rows_had != wide) return error.UnexpectedTensor;
                if (had) |h| if (h != rows_had) return error.UnexpectedTensor;
                had = rows_had;
                const w = try arena.alloc(u32, wide * d / per_word);
                const s = try arena.alloc(u16, wide * d / g);
                const b = try arena.alloc(u16, wide * d / g);
                @memcpy(w[0 .. own * d / per_word], words(parts[0].bytes)[0 .. own * d / per_word]);
                @memcpy(s[0 .. own * d / g], halves(parts[1].bytes)[0 .. own * d / g]);
                @memcpy(b[0 .. own * d / g], halves(parts[2].bytes)[0 .. own * d / g]);
                for (0..neurons) |k| {
                    const r = own + k;
                    const row = x[k * d ..][0..d];
                    const rw = w[r * d / per_word ..][0 .. d / per_word];
                    const rs = s[r * d / g ..][0 .. d / g];
                    const rb = b[r * d / g ..][0 .. d / g];
                    affine.quantize(q, row, rw, rs, rb);
                    affine.dequantize(q, rw, rs, rb, row);
                }
                bytes += try add(arena, edits, base, proj, .{ wide, d }, w, s, b, q);
            }
            // the output projection: each output row's own columns as they are, then a column a learned neuron
            var parts: [3]st.Tensor = undefined;
            for (&parts, [_][]const u8{ "weight", "scales", "biases" }) |*t, leaf| t.* = try files.get(try std.fmt.bufPrint(&name, "{s}{s}.{s}", .{ base, F.names.output, leaf }));
            const cols = had.?; // as wide as the input projections had rows
            const dw = try arena.alloc(u32, d * wide / per_word);
            const ds = try arena.alloc(u16, d * wide / g);
            const db = try arena.alloc(u16, d * wide / g);
            const col = try arena.alloc(f32, neurons);
            for (0..d) |o| {
                @memcpy(dw[o * wide / per_word ..][0 .. own / per_word], words(parts[0].bytes)[o * cols / per_word ..][0 .. own / per_word]);
                @memcpy(ds[o * wide / g ..][0 .. own / g], halves(parts[1].bytes)[o * cols / g ..][0 .. own / g]);
                @memcpy(db[o * wide / g ..][0 .. own / g], halves(parts[2].bytes)[o * cols / g ..][0 .. own / g]);
                for (col, 0..) |*x, k| x.* = v[k * d + o];
                const cw = dw[o * wide / per_word ..][own / per_word ..][0 .. neurons / per_word];
                const cs = ds[o * wide / g ..][own / g ..][0 .. neurons / g];
                const cb = db[o * wide / g ..][own / g ..][0 .. neurons / g];
                affine.quantize(q, col, cw, cs, cb);
                affine.dequantize(q, cw, cs, cb, col);
                for (col, 0..) |x, k| v[k * d + o] = x;
            }
            bytes += try add(arena, edits, base, F.names.output, .{ d, wide }, dw, ds, db, q);
            return bytes;
        }

        /// A projection's codes, scales and biases ([rows, cols] values) as three replacements; their bytes.
        fn add(arena: std.mem.Allocator, edits: *std.ArrayList(se.Replacement), base: []const u8, proj: []const u8, shape: [2]usize, w: []const u32, s: []const u16, b: []const u16, comptime q: family.Quant) !usize {
            const made = [_]struct { leaf: []const u8, dtype: st.DType, shape: [2]usize, bytes: []const u8 }{
                .{ .leaf = "weight", .dtype = .u32, .shape = .{ shape[0], shape[1] * q.bits / 32 }, .bytes = std.mem.sliceAsBytes(w) },
                .{ .leaf = "scales", .dtype = .bf16, .shape = .{ shape[0], shape[1] / q.group }, .bytes = std.mem.sliceAsBytes(s) },
                .{ .leaf = "biases", .dtype = .bf16, .shape = .{ shape[0], shape[1] / q.group }, .bytes = std.mem.sliceAsBytes(b) },
            };
            var bytes: usize = 0;
            for (made) |m| {
                try edits.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}{s}.{s}", .{ base, proj, m.leaf }), .dtype = m.dtype, .shape = try arena.dupe(usize, &m.shape), .bytes = m.bytes });
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
