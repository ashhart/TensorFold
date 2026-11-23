//! Tiny real safetensors exercise the production dry-run admission before any CUDA upload.
const std = @import("std");
const core = @import("core");
const admit = @import("cuda_exl3_admit.zig");
const t = std.testing;
const Case = struct { halves: usize = 2, marker: []const u8 = "I32", scale: []const u8 = "F16" };
fn write(dir: std.Io.Dir, c: Case) !void {
    const tn = 8 * 8 * c.halves * 8 * 2;
    const sn: usize = if (std.mem.eql(u8, c.scale, "F16")) 256 else 512;
    const header = try std.fmt.allocPrint(t.allocator, "{{\"p.trellis\":{{\"dtype\":\"I16\",\"shape\":[8,8,{d}],\"data_offsets\":[0,{d}]}}," ++
        "\"p.mul1\":{{\"dtype\":\"{s}\",\"shape\":[],\"data_offsets\":[{d},{d}]}}," ++
        "\"p.suh\":{{\"dtype\":\"{s}\",\"shape\":[128],\"data_offsets\":[{d},{d}]}}," ++
        "\"p.svh\":{{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[{d},{d}]}}}}", .{ c.halves * 8, tn, c.marker, tn, tn + 4, c.scale, tn + 4, tn + 4 + sn, tn + 4 + sn, tn + 4 + sn + 256 });
    defer t.allocator.free(header);
    const data = try t.allocator.alloc(u8, 8 + header.len + tn + 4 + sn + 256);
    defer t.allocator.free(data);
    @memset(data, 0);
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    std.mem.writeInt(u32, data[8 + header.len + tn ..][0..4], 0x83DCD12D, .little);
    try dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = data });
}
fn read(gpa: std.mem.Allocator, path: []const u8) !void {
    var ck = try core.Checkpoint.openModel(gpa, t.io, path);
    defer ck.close();
    _ = try admit.denseGroup(&ck, "p", 128, 128);
}
fn pathOf(tmp: anytype) ![]u8 {
    return std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}

test "EXL3 admission: invalid widths refuse before upload and dense admits only compiled variants" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    for ([_]usize{ 1, 17 }) |width| {
        try write(tmp.dir, .{ .halves = width });
        var ck = try core.Checkpoint.openModel(t.allocator, t.io, path);
        defer ck.close();
        try t.expectError(error.UnsupportedQwenQuantization, admit.group(&ck, "p", 128, 128));
    }
    try write(tmp.dir, .{ .halves = 15 });
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, path);
    defer ck.close();
    try t.expectEqual(@as(u32, 15), (try admit.group(&ck, "p", 128, 128)).halves);
    try t.expectError(error.UnsupportedQwenQuantization, admit.denseGroup(&ck, "p", 128, 128));
}

test "EXL3 admission: marker dtype and scale shape are checked on real mapped tensors" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    try write(tmp.dir, .{ .marker = "F32" });
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, path);
    defer ck.close();
    try t.expectError(error.UnsupportedQwenQuantization, admit.group(&ck, "p", 128, 128));
}

test "EXL3 admission: mixed supported widths and allocator failures stay host owned" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    for ([_]usize{ 2, 3, 8, 13, 14, 16 }) |width| {
        try write(tmp.dir, .{ .halves = width });
        try read(t.allocator, path);
    }
    // Match Checkpoint's own balanced-failure census, which tolerates asynchronous IO allocation counts.
    var reached = false;
    for (0..256) |i| {
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = i, .resize_fail_index = 0 });
        read(failing.allocator(), path) catch |err| if (!failing.has_induced_failure) return err;
        try t.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        if (!failing.has_induced_failure) {
            reached = true;
            break;
        }
    }
    try t.expect(reached);
    try write(tmp.dir, .{ .scale = "F32" });
    try t.expectError(error.UnexpectedTensor, read(t.allocator, path));
}

// --- Stage 6c: mcg codebook admission tests ---

const cuda_exl3 = @import("cuda").exl3;

fn writeMcg(dir: std.Io.Dir, halves: usize, magic: u32) !void {
    const tn = 8 * 8 * halves * 8 * 2;
    const sn: usize = 256;
    const header = try std.fmt.allocPrint(t.allocator, "{{\"p.trellis\":{{\"dtype\":\"I16\",\"shape\":[8,8,{d}],\"data_offsets\":[0,{d}]}}," ++
        "\"p.mcg\":{{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}," ++
        "\"p.suh\":{{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[{d},{d}]}}," ++
        "\"p.svh\":{{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[{d},{d}]}}}}", .{ halves * 8, tn, tn, tn + 4, tn + 4, tn + 4 + sn, tn + 4 + sn, tn + 4 + sn + 256 });
    defer t.allocator.free(header);
    const data = try t.allocator.alloc(u8, 8 + header.len + tn + 4 + sn + 256);
    defer t.allocator.free(data);
    @memset(data, 0);
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    std.mem.writeInt(u32, data[8 + header.len + tn ..][0..4], magic, .little);
    try dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = data });
}

test "EXL3 admission: an mcg-codebook group admits and reports codebook .mcg" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    try writeMcg(tmp.dir, 16, 0xCBAC1FED);
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, path);
    defer ck.close();
    const g = try admit.group(&ck, "p", 128, 128);
    try t.expectEqual(cuda_exl3.Codebook.mcg, g.codebook);
    try t.expectEqual(@as(u32, 16), g.halves);
    _ = try admit.denseGroup(&ck, "p", 128, 128);
}

test "EXL3 admission: an mcg marker with the wrong magic is refused" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    try writeMcg(tmp.dir, 16, 0xDEADBEEF);
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, path);
    defer ck.close();
    try t.expectError(error.UnsupportedQwenQuantization, admit.group(&ck, "p", 128, 128));
}

fn writeBoth(dir: std.Io.Dir, halves: usize) !void {
    const tn = 8 * 8 * halves * 8 * 2;
    const sn: usize = 256;
    const header = try std.fmt.allocPrint(t.allocator, "{{\"p.trellis\":{{\"dtype\":\"I16\",\"shape\":[8,8,{d}],\"data_offsets\":[0,{d}]}}," ++
        "\"p.mul1\":{{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}," ++
        "\"p.mcg\":{{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}," ++
        "\"p.suh\":{{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[{d},{d}]}}," ++
        "\"p.svh\":{{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[{d},{d}]}}}}", .{ halves * 8, tn, tn, tn + 4, tn + 4, tn + 8, tn + 8, tn + 8 + sn, tn + 8 + sn, tn + 8 + sn + 256 });
    defer t.allocator.free(header);
    const data = try t.allocator.alloc(u8, 8 + header.len + tn + 8 + sn + 256);
    defer t.allocator.free(data);
    @memset(data, 0);
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    std.mem.writeInt(u32, data[8 + header.len + tn ..][0..4], 0x83DCD12D, .little);
    std.mem.writeInt(u32, data[8 + header.len + tn + 4 ..][0..4], 0xCBAC1FED, .little);
    try dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = data });
}

test "EXL3 admission: a group carrying both mul1 and mcg markers is refused" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    try writeBoth(tmp.dir, 16);
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, path);
    defer ck.close();
    try t.expectError(error.UnsupportedQwenQuantization, admit.group(&ck, "p", 128, 128));
}
