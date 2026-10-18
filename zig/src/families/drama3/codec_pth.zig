//! codec.pth read without Python: the zip's stored members and a pickle VM limited to weights_only's allow-list.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const DType = enum { f64, f32, f16, bf16, i64, i32, i16, i8, u8, bool };

pub fn elementBytes(t: DType) usize {
    return switch (t) {
        .f64, .i64 => 8,
        .f32, .i32 => 4,
        .f16, .bf16, .i16 => 2,
        .i8, .u8, .bool => 1,
    };
}

/// torch's legacy storage class names, as torch.save pickles them.
fn storageType(name: []const u8) ?DType {
    const table = [_]struct { []const u8, DType }{
        .{ "DoubleStorage", .f64 },    .{ "FloatStorage", .f32 }, .{ "HalfStorage", .f16 },
        .{ "BFloat16Storage", .bf16 }, .{ "LongStorage", .i64 },  .{ "IntStorage", .i32 },
        .{ "ShortStorage", .i16 },     .{ "CharStorage", .i8 },   .{ "ByteStorage", .u8 },
        .{ "BoolStorage", .bool },
    };
    for (table) |e| if (std.mem.eql(u8, e[0], name)) return e[1];
    return null;
}

/// One state-dict entry: a view into the storage member `data/<storage>`.
pub const Tensor = struct {
    dtype: DType,
    storage: []const u8,
    storage_numel: u64,
    offset: u64, // elements
    shape: []const u64,
    stride: []const u64,
};

const Member = struct { local: u64, size: u64, crc: u32 };

pub const Archive = struct {
    arena: std.heap.ArenaAllocator,
    io: std.Io,
    file: std.Io.File,
    members: std.StringHashMapUnmanaged(Member),
    tensors: std.StringHashMapUnmanaged(Tensor),
    /// The archive's top directory with its slash ("<name>/"), as torch.save names it.
    prefix: []const u8,

    pub fn open(gpa: Allocator, io: std.Io, path: []const u8) !Archive {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const members = try readDirectory(a, io, file);
        var pickle_name: ?[]const u8 = null;
        var it = members.iterator();
        while (it.next()) |e| {
            if (!std.mem.endsWith(u8, e.key_ptr.*, "/data.pkl")) continue;
            if (pickle_name != null) return error.BadArchive;
            pickle_name = e.key_ptr.*;
        }
        const name = pickle_name orelse return error.BadArchive;
        const prefix = name[0 .. name.len - "data.pkl".len];
        const order_name = try std.mem.concat(a, u8, &.{ prefix, "byteorder" });
        if (members.get(order_name)) |m| {
            if (m.size > 16) return error.BadArchive;
            const order = try a.alloc(u8, @intCast(m.size));
            try readMember(io, file, m, order);
            if (!std.mem.eql(u8, order, "little")) return error.BigEndianArchive;
        }
        const pm = members.get(name).?;
        if (pm.size > 64 << 20) return error.BadArchive;
        const pickle = try a.alloc(u8, @intCast(pm.size));
        try readMember(io, file, pm, pickle);
        const tensors = try unpickle(a, pickle);
        return .{ .arena = arena, .io = io, .file = file, .members = members, .tensors = tensors, .prefix = prefix };
    }

    pub fn close(self: *Archive) void {
        self.file.close(self.io);
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn tensor(self: *const Archive, name: []const u8) !Tensor {
        return self.tensors.get(name) orelse {
            std.log.err("codec.pth has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    /// name's float32 elements row-major (caller frees), its shape and its storage member's CRC-32 checked.
    pub fn readF32(self: *const Archive, gpa: Allocator, name: []const u8, shape: []const u64) ![]f32 {
        const t = try self.tensor(name);
        if (t.dtype != .f32) {
            std.log.err("codec.pth tensor {s} is {s}, not float32", .{ name, @tagName(t.dtype) });
            return error.UnexpectedDType;
        }
        if (!std.mem.eql(u64, t.shape, shape)) {
            std.log.err("codec.pth tensor {s} has shape {any}, expected {any}", .{ name, t.shape, shape });
            return error.ShapeMismatch;
        }
        var numel: u64 = 1;
        var i = t.shape.len;
        while (i > 0) {
            i -= 1;
            if (t.shape[i] != 1 and t.stride[i] != numel) return error.NonContiguousTensor;
            numel *= t.shape[i];
        }
        if (t.offset > t.storage_numel or numel > t.storage_numel - t.offset) return error.BadArchive;
        const key = try std.fmt.allocPrint(gpa, "{s}data/{s}", .{ self.prefix, t.storage });
        defer gpa.free(key);
        const m = self.members.get(key) orelse return error.MissingStorage;
        if (m.size != t.storage_numel * 4) return error.BadArchive;
        const all = try gpa.alloc(f32, @intCast(t.storage_numel));
        errdefer gpa.free(all);
        try readMember(self.io, self.file, m, std.mem.sliceAsBytes(all));
        if (t.offset == 0 and numel == all.len) return all;
        const out = try gpa.dupe(f32, all[@intCast(t.offset)..][0..@intCast(numel)]);
        gpa.free(all);
        return out;
    }
};

fn readU16(b: []const u8, at: usize) u16 {
    return std.mem.readInt(u16, b[at..][0..2], .little);
}

fn readU32(b: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, b[at..][0..4], .little);
}

fn readU64(b: []const u8, at: usize) u64 {
    return std.mem.readInt(u64, b[at..][0..8], .little);
}

/// A stored member's bytes into `out` (exactly its size), CRC-checked.
fn readMember(io: std.Io, file: std.Io.File, m: Member, out: []u8) !void {
    if (out.len != m.size) return error.BadArchive;
    var head: [30]u8 = undefined;
    if (try file.readPositionalAll(io, &head, m.local) != head.len) return error.ShortRead;
    if (readU32(&head, 0) != 0x04034b50) return error.BadArchive;
    const data: u64 = m.local + @as(u64, head.len) + @as(u64, readU16(&head, 26)) + @as(u64, readU16(&head, 28));
    if (try file.readPositionalAll(io, out, data) != out.len) return error.ShortRead;
    if (std.hash.Crc32.hash(out) != m.crc) return error.ChecksumMismatch;
}

/// The central directory: every member stored (method 0), zip64 records honoured.
fn readDirectory(a: Allocator, io: std.Io, file: std.Io.File) !std.StringHashMapUnmanaged(Member) {
    const size = try file.length(io);
    if (size < 22) return error.BadArchive;
    const tail_len: usize = @intCast(@min(size, 22 + 0xFFFF));
    const tail = try a.alloc(u8, tail_len);
    if (try file.readPositionalAll(io, tail, size - @as(u64, tail_len)) != tail_len) return error.ShortRead;
    var at: usize = tail_len - 22;
    while (readU32(tail, at) != 0x06054b50) {
        if (at == 0) return error.BadArchive;
        at -= 1;
    }
    var count: u64 = readU16(tail, at + 10);
    var cd_size: u64 = readU32(tail, at + 12);
    var cd_off: u64 = readU32(tail, at + 16);
    if (count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF) {
        if (at < 20 or readU32(tail, at - 20) != 0x07064b50) return error.BadArchive;
        var rec: [56]u8 = undefined;
        if (try file.readPositionalAll(io, &rec, readU64(tail, at - 20 + 8)) != rec.len) return error.ShortRead;
        if (readU32(&rec, 0) != 0x06064b50) return error.BadArchive;
        count = readU64(&rec, 32);
        cd_size = readU64(&rec, 40);
        cd_off = readU64(&rec, 48);
    }
    if (cd_size > 64 << 20 or cd_off > size or cd_size > size - cd_off) return error.BadArchive;
    const cd = try a.alloc(u8, @intCast(cd_size));
    if (try file.readPositionalAll(io, cd, cd_off) != cd.len) return error.ShortRead;
    var members: std.StringHashMapUnmanaged(Member) = .empty;
    var p: usize = 0;
    var n: u64 = 0;
    while (n < count) : (n += 1) {
        if (cd.len - p < 46 or readU32(cd, p) != 0x02014b50) return error.BadArchive;
        const method = readU16(cd, p + 10);
        const crc = readU32(cd, p + 16);
        var csize: u64 = readU32(cd, p + 20);
        var usize_: u64 = readU32(cd, p + 24);
        const nlen: usize = readU16(cd, p + 28);
        const xlen: usize = readU16(cd, p + 30);
        const clen: usize = readU16(cd, p + 32);
        var local: u64 = readU32(cd, p + 42);
        if (cd.len - p - 46 < nlen + xlen + clen) return error.BadArchive;
        const name = cd[p + 46 ..][0..nlen];
        var x = p + 46 + nlen;
        const xend = x + xlen;
        while (xend - x >= 4) {
            const id = readU16(cd, x);
            const len: usize = readU16(cd, x + 2);
            if (xend - x - 4 < len) return error.BadArchive;
            if (id == 0x0001) {
                var f = x + 4;
                const fend = f + len;
                if (usize_ == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    usize_ = readU64(cd, f);
                    f += 8;
                }
                if (csize == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    csize = readU64(cd, f);
                    f += 8;
                }
                if (local == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    local = readU64(cd, f);
                }
            }
            x += 4 + len;
        }
        if (method != 0 or csize != usize_) {
            std.log.err("codec.pth member {s} is compressed; torch.save stores members", .{name});
            return error.CompressedMember;
        }
        if (local > size or usize_ > size - local) return error.BadArchive;
        try members.put(a, name, .{ .local = local, .size = usize_, .crc = crc });
        p += 46 + nlen + xlen + clen;
    }
    return members;
}

const GlobalRef = union(enum) { ordered_dict, rebuild_tensor_v2, storage_type: DType };

const Storage = struct { dtype: DType, key: []const u8, numel: u64 };

const Dict = struct {
    keys: std.ArrayList(Value) = .empty,
    values: std.ArrayList(Value) = .empty,

    fn put(d: *Dict, a: Allocator, k: Value, v: Value) !void {
        try d.keys.append(a, k);
        try d.values.append(a, v);
    }
};

const Value = union(enum) {
    none,
    boolean: bool,
    int: i64,
    str: []const u8,
    tuple: []const Value,
    dict: *Dict,
    global: GlobalRef,
    storage: Storage,
    tensor: *const Tensor,
};

const Reader = struct {
    data: []const u8,
    at: usize = 0,

    fn bytes(r: *Reader, n: usize) ![]const u8 {
        if (r.data.len - r.at < n) return error.BadPickle;
        defer r.at += n;
        return r.data[r.at..][0..n];
    }

    fn byte(r: *Reader) !u8 {
        return (try r.bytes(1))[0];
    }

    fn int(r: *Reader, comptime T: type) !T {
        return std.mem.readInt(T, (try r.bytes(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn line(r: *Reader) ![]const u8 {
        const end = std.mem.indexOfScalarPos(u8, r.data, r.at, '\n') orelse return error.BadPickle;
        defer r.at = end + 1;
        return r.data[r.at..end];
    }
};

fn resolveGlobal(module: []const u8, name: []const u8) !GlobalRef {
    if (std.mem.eql(u8, module, "collections") and std.mem.eql(u8, name, "OrderedDict")) return .ordered_dict;
    if (std.mem.eql(u8, module, "torch._utils") and std.mem.eql(u8, name, "_rebuild_tensor_v2")) return .rebuild_tensor_v2;
    if (std.mem.eql(u8, module, "torch")) if (storageType(name)) |t| return .{ .storage_type = t };
    std.log.warn("codec.pth pickle names {s}.{s}, outside the tensor state-dict allow-list", .{ module, name });
    return error.ForbiddenGlobal;
}

fn pop(stack: *std.ArrayList(Value)) !Value {
    return stack.pop() orelse error.BadPickle;
}

fn topDict(stack: *std.ArrayList(Value)) !*Dict {
    if (stack.items.len == 0) return error.BadPickle;
    return switch (stack.items[stack.items.len - 1]) {
        .dict => |d| d,
        else => error.BadPickle,
    };
}

fn newDict(a: Allocator) !*Dict {
    const d = try a.create(Dict);
    d.* = .{};
    return d;
}

fn nonNegative(v: Value) !u64 {
    return switch (v) {
        .int => |i| if (i < 0) error.BadPickle else @intCast(i),
        else => error.BadPickle,
    };
}

fn intTuple(a: Allocator, v: Value) ![]const u64 {
    const items = switch (v) {
        .tuple => |t| t,
        else => return error.BadPickle,
    };
    const out = try a.alloc(u64, items.len);
    for (items, out) |item, *o| o.* = try nonNegative(item);
    return out;
}

/// torch.save's persistent id for a storage: ("storage", <Type>Storage, key, location, numel).
fn persistent(pid: Value) !Storage {
    const t = switch (pid) {
        .tuple => |t| t,
        else => return error.BadPickle,
    };
    if (t.len != 5) return error.BadPickle;
    if (t[0] != .str or !std.mem.eql(u8, t[0].str, "storage")) return error.BadPickle;
    const dtype = switch (t[1]) {
        .global => |g| switch (g) {
            .storage_type => |d| d,
            else => return error.BadPickle,
        },
        else => return error.BadPickle,
    };
    if (t[2] != .str or t[3] != .str) return error.BadPickle;
    return .{ .dtype = dtype, .key = t[2].str, .numel = try nonNegative(t[4]) };
}

fn reduce(a: Allocator, f: Value, args: Value) !Value {
    const items = switch (args) {
        .tuple => |t| t,
        else => return error.BadPickle,
    };
    const g = switch (f) {
        .global => |g| g,
        else => return error.BadPickle,
    };
    switch (g) {
        .ordered_dict => {
            if (items.len != 0) return error.BadPickle;
            return .{ .dict = try newDict(a) };
        },
        .rebuild_tensor_v2 => {
            // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
            if (items.len < 6 or items.len > 7) return error.BadPickle;
            const st = switch (items[0]) {
                .storage => |s| s,
                else => return error.BadPickle,
            };
            const shape = try intTuple(a, items[2]);
            const stride = try intTuple(a, items[3]);
            if (shape.len != stride.len) return error.BadPickle;
            if (items[4] != .boolean) return error.BadPickle;
            switch (items[5]) {
                .dict => |d| if (d.keys.items.len != 0) return error.BadPickle,
                else => return error.BadPickle,
            }
            const t = try a.create(Tensor);
            t.* = .{ .dtype = st.dtype, .storage = st.key, .storage_numel = st.numel, .offset = try nonNegative(items[1]), .shape = shape, .stride = stride };
            return .{ .tensor = t };
        },
        .storage_type => return error.BadPickle,
    }
}

/// The state dict's tensors by name. Unknown opcodes and globals are refused, never run.
fn unpickle(a: Allocator, data: []const u8) !std.StringHashMapUnmanaged(Tensor) {
    var stack: std.ArrayList(Value) = .empty;
    var marks: std.ArrayList(usize) = .empty;
    var memo: std.AutoHashMapUnmanaged(u32, Value) = .empty;
    var r: Reader = .{ .data = data };
    while (true) {
        const op = try r.byte();
        switch (op) {
            0x80 => { // PROTO
                const v = try r.byte();
                if (v < 2 or v > 5) return error.UnsupportedPickle;
            },
            '}' => try stack.append(a, .{ .dict = try newDict(a) }), // EMPTY_DICT
            ')' => try stack.append(a, .{ .tuple = &.{} }), // EMPTY_TUPLE
            '(' => try marks.append(a, stack.items.len), // MARK
            't' => { // TUPLE
                const at = marks.pop() orelse return error.BadPickle;
                if (at > stack.items.len) return error.BadPickle;
                const items = try a.dupe(Value, stack.items[at..]);
                stack.shrinkRetainingCapacity(at);
                try stack.append(a, .{ .tuple = items });
            },
            0x85, 0x86, 0x87 => { // TUPLE1, TUPLE2, TUPLE3
                const n: usize = op - 0x84;
                if (stack.items.len < n) return error.BadPickle;
                const at = stack.items.len - n;
                const items = try a.dupe(Value, stack.items[at..]);
                stack.shrinkRetainingCapacity(at);
                try stack.append(a, .{ .tuple = items });
            },
            'X' => try stack.append(a, .{ .str = try r.bytes(try r.int(u32)) }), // BINUNICODE
            0x8c => try stack.append(a, .{ .str = try r.bytes(try r.byte()) }), // SHORT_BINUNICODE
            'c' => { // GLOBAL
                const module = try r.line();
                const name = try r.line();
                try stack.append(a, .{ .global = try resolveGlobal(module, name) });
            },
            'K' => try stack.append(a, .{ .int = try r.byte() }), // BININT1
            'M' => try stack.append(a, .{ .int = try r.int(u16) }), // BININT2
            'J' => try stack.append(a, .{ .int = try r.int(i32) }), // BININT
            0x8a => { // LONG1
                const n = try r.byte();
                if (n > 8) return error.UnsupportedPickle;
                const b = try r.bytes(n);
                var v: i64 = 0;
                if (n > 0) {
                    var buf: [8]u8 = if (b[n - 1] & 0x80 != 0) @splat(0xFF) else @splat(0);
                    @memcpy(buf[0..n], b);
                    v = std.mem.readInt(i64, &buf, .little);
                }
                try stack.append(a, .{ .int = v });
            },
            'N' => try stack.append(a, .none),
            0x88 => try stack.append(a, .{ .boolean = true }), // NEWTRUE
            0x89 => try stack.append(a, .{ .boolean = false }), // NEWFALSE
            'q', 'r' => { // BINPUT, LONG_BINPUT
                const key: u32 = if (op == 'q') try r.byte() else try r.int(u32);
                if (stack.items.len == 0) return error.BadPickle;
                try memo.put(a, key, stack.items[stack.items.len - 1]);
            },
            'h', 'j' => { // BINGET, LONG_BINGET
                const key: u32 = if (op == 'h') try r.byte() else try r.int(u32);
                try stack.append(a, memo.get(key) orelse return error.BadPickle);
            },
            'Q' => try stack.append(a, .{ .storage = try persistent(try pop(&stack)) }), // BINPERSID
            'R' => { // REDUCE
                const args = try pop(&stack);
                const f = try pop(&stack);
                try stack.append(a, try reduce(a, f, args));
            },
            's' => { // SETITEM
                const v = try pop(&stack);
                const k = try pop(&stack);
                try (try topDict(&stack)).put(a, k, v);
            },
            'u' => { // SETITEMS
                const at = marks.pop() orelse return error.BadPickle;
                if (at == 0 or at > stack.items.len) return error.BadPickle;
                const d = switch (stack.items[at - 1]) {
                    .dict => |d| d,
                    else => return error.BadPickle,
                };
                const items = stack.items[at..];
                if (items.len % 2 != 0) return error.BadPickle;
                var i: usize = 0;
                while (i < items.len) : (i += 2) try d.put(a, items[i], items[i + 1]);
                stack.shrinkRetainingCapacity(at);
            },
            'b' => { // BUILD: the state dict's _metadata attribute, not needed to read tensors
                switch (try pop(&stack)) {
                    .dict, .none => {},
                    else => return error.BadPickle,
                }
                _ = try topDict(&stack);
            },
            '.' => { // STOP
                const root = switch (try pop(&stack)) {
                    .dict => |d| d,
                    else => return error.BadPickle,
                };
                var out: std.StringHashMapUnmanaged(Tensor) = .empty;
                for (root.keys.items, root.values.items) |k, v| {
                    if (k != .str or v != .tensor) return error.BadPickle;
                    const gop = try out.getOrPut(a, k.str);
                    if (gop.found_existing) return error.BadPickle;
                    gop.value_ptr.* = v.tensor.*;
                }
                return out;
            },
            else => {
                std.log.warn("codec.pth pickle opcode 0x{x:0>2} is outside the state-dict subset", .{op});
                return error.UnsupportedPickle;
            },
        }
    }
}

test "the state-dict VM reads torch.save's tensor records and their views" {
    // A protocol-2 state-dict pickle as torch.save writes it, built by a Python pickler without torch.
    const fixture = [_]u8{
        0x80, 0x02, 0x63, 0x63, 0x6f, 0x6c, 0x6c, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x0a, 0x4f,
        0x72, 0x64, 0x65, 0x72, 0x65, 0x64, 0x44, 0x69, 0x63, 0x74, 0x0a, 0x71, 0x00, 0x29, 0x52, 0x71,
        0x01, 0x28, 0x58, 0x08, 0x00, 0x00, 0x00, 0x61, 0x2e, 0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x71,
        0x02, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68, 0x2e, 0x5f, 0x75, 0x74, 0x69, 0x6c, 0x73, 0x0a, 0x5f,
        0x72, 0x65, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x5f, 0x76,
        0x32, 0x0a, 0x71, 0x03, 0x28, 0x28, 0x58, 0x07, 0x00, 0x00, 0x00, 0x73, 0x74, 0x6f, 0x72, 0x61,
        0x67, 0x65, 0x71, 0x04, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68, 0x0a, 0x46, 0x6c, 0x6f, 0x61, 0x74,
        0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x0a, 0x71, 0x05, 0x58, 0x01, 0x00, 0x00, 0x00, 0x30,
        0x71, 0x06, 0x58, 0x03, 0x00, 0x00, 0x00, 0x63, 0x70, 0x75, 0x71, 0x07, 0x4b, 0x06, 0x74, 0x71,
        0x08, 0x51, 0x4b, 0x00, 0x4b, 0x02, 0x4b, 0x03, 0x86, 0x71, 0x09, 0x4b, 0x03, 0x4b, 0x01, 0x86,
        0x71, 0x0a, 0x89, 0x68, 0x00, 0x29, 0x52, 0x71, 0x0b, 0x74, 0x71, 0x0c, 0x52, 0x71, 0x0d, 0x58,
        0x06, 0x00, 0x00, 0x00, 0x61, 0x2e, 0x62, 0x69, 0x61, 0x73, 0x71, 0x0e, 0x68, 0x03, 0x28, 0x28,
        0x68, 0x04, 0x68, 0x05, 0x68, 0x06, 0x68, 0x07, 0x4b, 0x06, 0x74, 0x71, 0x0f, 0x51, 0x4b, 0x04,
        0x4b, 0x02, 0x85, 0x71, 0x10, 0x4b, 0x01, 0x85, 0x71, 0x11, 0x89, 0x68, 0x00, 0x29, 0x52, 0x71,
        0x12, 0x74, 0x71, 0x13, 0x52, 0x71, 0x14, 0x58, 0x01, 0x00, 0x00, 0x00, 0x62, 0x71, 0x15, 0x68,
        0x03, 0x28, 0x28, 0x68, 0x04, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68, 0x0a, 0x42, 0x46, 0x6c, 0x6f,
        0x61, 0x74, 0x31, 0x36, 0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x0a, 0x71, 0x16, 0x58, 0x01,
        0x00, 0x00, 0x00, 0x31, 0x71, 0x17, 0x68, 0x07, 0x4d, 0x2c, 0x01, 0x74, 0x71, 0x18, 0x51, 0x4b,
        0x00, 0x4d, 0x2c, 0x01, 0x85, 0x71, 0x19, 0x68, 0x11, 0x89, 0x68, 0x00, 0x29, 0x52, 0x71, 0x1a,
        0x74, 0x71, 0x1b, 0x52, 0x71, 0x1c, 0x75, 0x7d, 0x71, 0x1d, 0x58, 0x09, 0x00, 0x00, 0x00, 0x5f,
        0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x71, 0x1e, 0x68, 0x00, 0x29, 0x52, 0x71, 0x1f,
        0x28, 0x58, 0x00, 0x00, 0x00, 0x00, 0x71, 0x20, 0x7d, 0x71, 0x21, 0x58, 0x07, 0x00, 0x00, 0x00,
        0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x71, 0x22, 0x4b, 0x01, 0x73, 0x58, 0x01, 0x00, 0x00,
        0x00, 0x61, 0x71, 0x23, 0x7d, 0x71, 0x24, 0x68, 0x22, 0x4b, 0x02, 0x73, 0x75, 0x73, 0x62, 0x2e,
    };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const tensors = try unpickle(arena.allocator(), &fixture);
    try std.testing.expectEqual(@as(u32, 3), tensors.count());
    const w = tensors.get("a.weight").?;
    try std.testing.expectEqual(DType.f32, w.dtype);
    try std.testing.expectEqualStrings("0", w.storage);
    try std.testing.expectEqual(@as(u64, 6), w.storage_numel);
    try std.testing.expectEqualSlices(u64, &.{ 2, 3 }, w.shape);
    try std.testing.expectEqualSlices(u64, &.{ 3, 1 }, w.stride);
    const b = tensors.get("a.bias").?;
    try std.testing.expectEqual(@as(u64, 4), b.offset);
    try std.testing.expectEqualSlices(u64, &.{2}, b.shape);
    const c = tensors.get("b").?;
    try std.testing.expectEqual(DType.bf16, c.dtype);
    try std.testing.expectEqual(@as(u64, 300), c.storage_numel);
    try std.testing.expectEqualSlices(u64, &.{300}, c.shape);
}

test "the state-dict VM refuses globals outside the allow-list" {
    // PROTO 2; GLOBAL os.system; STOP
    const evil = "\x80\x02cos\nsystem\n.";
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.ForbiddenGlobal, unpickle(arena.allocator(), evil));
}
