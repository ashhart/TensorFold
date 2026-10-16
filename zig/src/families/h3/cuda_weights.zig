//! The H3 CUDA family's block weights: quantized to int8 on the GPU from the checkpoint, or read from the saved copy.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const Checkpoint = @import("checkpoint.zig").Checkpoint;
const Model = h3.Model;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, offset: i64) isize;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;
extern "c" fn rename(from: [*:0]const u8, to: [*:0]const u8) c_int;

const staging_bytes = 64 << 20;

/// What the int8 copy starts with: its format, the checkpoint's size and the model's sizes.
const CacheHeader = extern struct { magic: [8]u8 = "tfh3i8\x00\x01".*, source_bytes: u64, hidden: u32, inner: u32, mlp: u32, layers: u32 };

/// Reads the model's sizes from `path`, allocates its blocks and fills them; true when the saved int8 copy was used.
pub fn read(self: *Model, path: [*:0]const u8) !bool {
    const d = &self.driver;
    var staging = try cuda.HostBuffer.alloc(d, staging_bytes);
    defer staging.free();
    var ck = try Checkpoint.open(path);
    defer ck.deinit();
    var cache_path: [1024]u8 = undefined;
    const cache: [*:0]const u8 = @ptrCast((try std.fmt.bufPrint(&cache_path, "{s}.tf-int8\x00", .{std.mem.span(path)})).ptr);
    var name: [128]u8 = undefined;
    const qkv0 = try ck.bf16("blocks.0.attn.qkv_proj.weight");
    const fc2_0 = try ck.bf16("blocks.0.mlp.fc2.weight");
    self.hidden = qkv0.cols;
    self.inner = qkv0.rows / 3;
    self.heads = self.inner / h3.head_dim;
    self.mlp = fc2_0.cols;
    var layers: usize = 0;
    while (std.mem.indexOf(u8, ck.header, try std.fmt.bufPrint(&name, "\"blocks.{d}.norm1.weight\"", .{layers})) != null) layers += 1;
    const hidden = self.hidden;
    const inner = self.inner;

    self.blocks = try self.gpa.alloc(h3.Block, layers);
    for (self.blocks) |*b| {
        b.qkvc = try quantAlloc(self, 4 * inner, hidden);
        b.out = try quantAlloc(self, hidden, inner);
        b.fc1 = try quantAlloc(self, 2 * self.mlp, hidden);
        b.fc2 = try quantAlloc(self, hidden, self.mlp);
        b.norms = try cuda.DeviceBuffer.alloc(d, (2 * hidden + 2 * h3.head_dim) * 2);
    }
    if (readCache(self, &staging, cache, ck.bytes.len) catch false) return true;
    var scratch = try cuda.DeviceBuffer.alloc(d, 4 * inner * hidden * 2);
    defer scratch.free();
    for (self.blocks, 0..) |*b, i| {
        try quantize(self, &staging, &scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.qkv_proj.weight", .{i})), b.qkvc, 0, 0);
        try quantize(self, &staging, &scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.to_gate_compress.weight", .{i})), b.qkvc, 3 * inner, 0);
        try quantize(self, &staging, &scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.out_proj.weight", .{i})), b.out, 0, 0);
        // The SwiGLU's first projection is stored [gate; value]; its rows are laid out value, gate, value, gate.
        try quantize(self, &staging, &scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.mlp.fc1.weight", .{i})), b.fc1, 0, self.mlp);
        try quantize(self, &staging, &scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.mlp.fc2.weight", .{i})), b.fc2, 0, 0);
        try upload(&staging, b.norms, 0, (try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.norm1.weight", .{i}))).bytes);
        try upload(&staging, b.norms, hidden * 2, (try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.norm2.weight", .{i}))).bytes);
        try upload(&staging, b.norms, hidden * 4, (try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.q_norm.weight", .{i}))).bytes);
        try upload(&staging, b.norms, hidden * 4 + h3.head_dim * 2, (try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.k_norm.weight", .{i}))).bytes);
    }
    writeCache(self, cache, ck.bytes.len) catch |e| std.debug.print("tensorfold h3: the int8 copy was not saved ({t}); the next start converts again\n", .{e});
    return false;
}

fn quantAlloc(self: *Model, n: usize, k: usize) !h3.Quant {
    return .{ .w = try cuda.DeviceBuffer.alloc(&self.driver, n * k), .s = try cuda.DeviceBuffer.alloc(&self.driver, n * 4) };
}

/// `bytes` into `to` from `offset` through pinned memory; the driver copies pageable memory several times slower.
fn upload(staging: *cuda.HostBuffer, to: cuda.DeviceBuffer, offset: usize, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = @min(staging_bytes, bytes.len - done);
        @memcpy(staging.bytes[0..n], bytes[done..][0..n]);
        try to.upload(offset + done, staging.bytes[0..n]);
        done += n;
    }
}

/// `w` stored (N, K) in bf16 to int8 rows of `q` from `row0`, or interleaved around `half`.
fn quantize(self: *Model, staging: *cuda.HostBuffer, scratch: *cuda.DeviceBuffer, w: Checkpoint.Entry, q: h3.Quant, row0: usize, half: usize) !void {
    try upload(staging, scratch.*, 0, w.bytes);
    var args: cuda.Args = .{};
    args.add(scratch.ptr);
    args.add(q.w.ptr);
    args.add(q.s.ptr);
    args.add(@as(c_int, @intCast(w.cols)));
    args.add(@as(c_int, @intCast(row0)));
    args.add(@as(c_int, @intCast(half)));
    try self.launch("h3_quant_weight", .{ .x = @intCast(w.rows) }, 128, 0, &args);
    try self.stream.synchronize();
}

fn cacheBuffers(b: *h3.Block) [9]*cuda.DeviceBuffer {
    return .{ &b.qkvc.w, &b.qkvc.s, &b.out.w, &b.out.s, &b.fc1.w, &b.fc1.s, &b.fc2.w, &b.fc2.s, &b.norms };
}

fn header(self: *const Model, source_bytes: usize) CacheHeader {
    return .{ .source_bytes = source_bytes, .hidden = @intCast(self.hidden), .inner = @intCast(self.inner), .mlp = @intCast(self.mlp), .layers = @intCast(self.blocks.len) };
}

/// The int8 projections from an earlier load's copy beside the checkpoint; false when there is none that fits.
fn readCache(self: *Model, staging: *cuda.HostBuffer, path: [*:0]const u8, source_bytes: usize) !bool {
    const fd = open(path, 0);
    if (fd < 0) return false;
    defer _ = close(fd);
    const want = header(self, source_bytes);
    var total: usize = @sizeOf(CacheHeader);
    for (self.blocks) |*b| for (cacheBuffers(b)) |held| {
        total += held.len;
    };
    if (lseek(fd, 0, 2) != @as(i64, @intCast(total))) return false;
    var found: CacheHeader = undefined;
    if (pread(fd, @ptrCast(&found), @sizeOf(CacheHeader), 0) != @sizeOf(CacheHeader)) return false;
    if (!std.mem.eql(u8, std.mem.asBytes(&found), std.mem.asBytes(&want))) return false;
    var at: usize = @sizeOf(CacheHeader);
    for (self.blocks) |*b| for (cacheBuffers(b)) |held| {
        var done: usize = 0;
        while (done < held.len) {
            const n = @min(staging_bytes, held.len - done);
            if (pread(fd, staging.bytes.ptr, n, @intCast(at + done)) != @as(isize, @intCast(n))) return error.CannotRead;
            try held.upload(done, staging.bytes[0..n]);
            done += n;
        }
        at += held.len;
    };
    return true;
}

/// Written under a temporary name and renamed, so a stopped write leaves no half file to read.
fn writeCache(self: *Model, path: [*:0]const u8, source_bytes: usize) !void {
    var tmp_path: [1040]u8 = undefined;
    const tmp: [*:0]const u8 = @ptrCast((try std.fmt.bufPrint(&tmp_path, "{s}.part\x00", .{std.mem.span(path)})).ptr);
    const file = fopen(tmp, "wb") orelse return error.CannotWrite;
    var open_file = true;
    defer if (open_file) {
        _ = fclose(file);
    };
    const head = header(self, source_bytes);
    if (fwrite(&head, 1, @sizeOf(CacheHeader), file) != @sizeOf(CacheHeader)) return error.CannotWrite;
    const host = try self.gpa.alloc(u8, 4 * self.inner * self.hidden);
    defer self.gpa.free(host);
    for (self.blocks) |*b| for (cacheBuffers(b)) |held| {
        try held.download(0, host[0..held.len]);
        if (fwrite(host.ptr, 1, held.len, file) != held.len) return error.CannotWrite;
    };
    open_file = false;
    if (fclose(file) != 0) return error.CannotWrite;
    if (rename(tmp, path) != 0) return error.CannotWrite;
}

/// Frees what `read` allocated.
pub fn free(self: *Model) void {
    for (self.blocks) |*b| for (cacheBuffers(b)) |held| held.free();
    self.gpa.free(self.blocks);
}
