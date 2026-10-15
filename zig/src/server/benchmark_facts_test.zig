//! Checkpoint facts and the hardware parsers on fictional inputs.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const model = @import("benchmark_model.zig");
const host = @import("benchmark_host.zig");
const receipt = @import("benchmark_receipt.zig");
const fake = @import("benchmark_fake.zig");
const check = @import("benchmark_check.zig");
const suite = @import("benchmark_suite.zig");

const revision = "0123456789abcdef0123456789abcdef01234567";
const families = [_]api.Family{.{ .model_type = "qwen3_5", .formats = &.{} }};
const config = "{\"model_type\": \"qwen3_5\", \"quantization\": {\"bits\": 4, \"group_size\": 64}, \"_name_or_path\": \"/home/example/private-checkpoint\"}";

fn checkpoint(a: std.mem.Allocator, root: []const u8, parts: []const []const u8, config_text: []const u8) ![]const u8 {
    const dir = try std.fs.path.join(a, try std.mem.concat(a, []const u8, &.{ &.{root}, parts }));
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(std.testing.io, dir);
    try cwd.writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = config_text });
    try cwd.writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "tokenizer.json" }), .data = "{\"version\":\"1.0\"}" });
    return dir;
}

fn modelJson(a: std.mem.Allocator, m: model.Model) ![]const u8 {
    const v = try receipt.value(a, "x", "x", m, .{ .tensorfold_version = "x", .backend = "mlx", .python_version = "none", .platform = "macos" }, .{}, .{ .tokens = 2, .repetitions = 1, .temperatures = &.{0}, .serial = false, .context_tokens = null, .managed = true }, &.{});
    return json.stringify(a, v.get("model").?, .{});
}

test "a local checkpoint yields hashes and labels but never its name or path" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try fake.tmpPath(a, &tmp);
    const dir = try checkpoint(a, root, &.{"private-checkpoint"}, config);
    const m = try model.describe(a, std.testing.io, dir, dir, true, &families, null);
    try std.testing.expect(m.repo_id == null and m.revision == null and m.local);
    try std.testing.expectEqualStrings("qwen3_5", m.family.?);
    try std.testing.expectEqualStrings("mlx-4bit-g64", m.quantization.?);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(config, &d, .{});
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(d, .lower), m.config_sha256.?);
    try std.testing.expect(m.tokenizer_sha256 != null);
    const text = try modelJson(a, m);
    for ([_][]const u8{ root, "private-checkpoint", "_name_or_path" }) |leak| try std.testing.expect(std.mem.indexOf(u8, text, leak) == null);
}

test "a cached repo takes its revision from the snapshot directory; a drafter repo is named, a drafter path is not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try fake.tmpPath(a, &tmp);
    const dir = try checkpoint(a, root, &.{ "snapshots", revision }, config);
    const m = try model.describe(a, std.testing.io, "example/model", dir, false, &families, "example/draft");
    try std.testing.expect(!m.local and std.mem.eql(u8, m.repo_id.?, "example/model") and std.mem.eql(u8, m.revision.?, revision));
    try std.testing.expectEqualStrings("example/draft", m.drafter_repo_id.?);
    try std.testing.expect((try model.describe(a, std.testing.io, "example/model", dir, false, &.{}, dir)).drafter_repo_id == null);
    try std.testing.expect((try model.describe(a, std.testing.io, "example/model", dir, false, &.{}, null)).family == null);
}

test "a known drafter is named only for a target of its family and shape" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { config: []const u8, drafter: ?[]const u8 }{
        .{ .config = "{\"model_type\": \"qwen3_5\", \"hidden_size\": 5120, \"num_hidden_layers\": 64}", .drafter = "z-lab/Qwen3.8-27B-DFlash2" },
        .{ .config = "{\"text_config\": {\"model_type\": \"qwen3_5\", \"hidden_size\": 5120, \"num_hidden_layers\": 64}}", .drafter = "z-lab/Qwen3.8-27B-DFlash2" },
        .{ .config = "{\"model_type\": \"qwen3_5\", \"hidden_size\": 2048, \"num_hidden_layers\": 40}", .drafter = null },
        .{ .config = "{\"model_type\": \"qwen3_5\", \"hidden_size\": 5120}", .drafter = null },
        .{ .config = "{\"model_type\": \"nemotron_h\", \"hidden_size\": 5120, \"num_hidden_layers\": 64}", .drafter = null },
        .{ .config = "[]", .drafter = null },
    };
    for (cases) |c| {
        const got = model.knownDrafter((try json.parse(a, c.config)).ok);
        if (c.drafter) |d| try std.testing.expectEqualStrings(d, got.?) else try std.testing.expect(got == null);
    }
}

test "the tokenizer hash covers the chat template and not the weights; limits give nulls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const dir = try checkpoint(a, try fake.tmpPath(a, &tmp), &.{"target"}, config);
    const before = model.tokenizerHash(a, io, dir).?;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "model.safetensors" }), .data = "weights would not be read" });
    try std.testing.expectEqualStrings(before, model.tokenizerHash(a, io, dir).?);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "chat_template.jinja" }), .data = "{{ messages }}" });
    try std.testing.expect(!std.mem.eql(u8, before, model.tokenizerHash(a, io, dir).?));
    const big = try a.alloc(u8, (4 << 20) + 1);
    @memset(big, ' ');
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = big });
    const m = try model.describe(a, io, dir, dir, true, &families, null);
    try std.testing.expect(m.config_sha256 == null and m.family == null and m.quantization == null);
}

test "quantization labels come only from the allowlist" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { []const u8, ?[]const u8 }{
        .{ config, "mlx-4bit-g64" },
        .{ "{\"text_config\": {\"quantization_config\": {\"quant_method\": \"ModelOpt\", \"quant_algo\": \"NVFP4\", \"group_size\": 16}}}", "modelopt-nvfp4-g16" },
        .{ "{\"model_type\": \"x\"}", "unquantized" },
        .{ "{}", null },
        .{ "{\"model_type\": \"private-family\", \"quantization_config\": {\"quant_method\": \"private-format\", \"quant_algo\": \"/home/example/private\", \"bits\": \"secret\"}}", null },
        .{ "{\"quantization\": {\"quant_method\": \"awq\", \"mode\": \"/home/x\", \"bits\": true, \"group_size\": 99999}}", "awq" },
    }) |c| {
        const got = try model.quantization(a, (try json.parse(a, c[0])).ok);
        if (c[1]) |want| try std.testing.expectEqualStrings(want, got.?) else try std.testing.expect(got == null);
    }
}

test "hardware names pass only the allowlist; Mac and NVIDIA probes keep only names, memory and cores" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("Intel(R) Xeon(R) CPU E5-2680 v4", host.modelName(a, "  Intel(R)  Xeon(R) CPU E5-2680 v4 @ 2.40GHz\n").?);
    try std.testing.expectEqualStrings("Apple M5 Max", host.modelName(a, "Apple M5 Max").?);
    const long = try std.mem.concat(a, u8, &.{ "NVIDIA ", try a.dupe(u8, &(@as([121]u8, @splat('a')))) });
    for ([_][]const u8{ "private-machine", "/home/example/private", "NVIDIA /home/example/private", long, "Apple M5 Max Extra", "Apple M", "Intel user@example.test" }) |bad| try std.testing.expect(host.modelName(a, bad) == null);
    const gib128: u64 = 128 << 30;
    const profiler = "{\"SPDisplaysDataType\": [{\"sppci_model\": \"Apple M5 Max\", \"sppci_cores\": \"40\", \"serial_number\": \"private-device-value\", \"_name\": \"private-computer\", \"sppci_device_id\": \"private-id\"}]}";
    const mac = host.summarize("Apple M5 Max", gib128, host.macGpus(a, profiler, gib128), true);
    try std.testing.expectEqualStrings("detected", mac.source);
    try std.testing.expectEqualStrings("unified", mac.memory_type);
    try std.testing.expect(mac.gpus.len == 1 and mac.gpus[0].cores.? == 40 and mac.gpus[0].memory_bytes.? == gib128);
    try std.testing.expect(host.macGpus(a, "{bad json", null).len == 0 and host.macGpus(a, "{\"SPDisplaysDataType\": \"not-a-list\"}", null).len == 0);
    const smi = "NVIDIA RTX 4090, 24564, 580.65.06\nNVIDIA GB10, N/A, 580.65.06\n";
    const rows = host.nvidiaRows(a, smi, null);
    try std.testing.expect(rows.len == 2 and rows[0].memory_bytes.? == 24564 << 20 and rows[1].memory_bytes == null);
    try std.testing.expectEqualStrings("580.65.06", host.driver(rows).?);
    const gpus = [_]host.Gpu{ .{ .name = rows[0].name, .memory_bytes = rows[0].memory_bytes }, .{ .name = rows[1].name } };
    const cuda = host.summarize("AMD EPYC 9004", 256 << 30, &gpus, false);
    try std.testing.expect(std.mem.eql(u8, cuda.memory_type, "dedicated") and std.mem.eql(u8, cuda.source, "partial"));
    const one = host.nvidiaRows(a, smi, "1");
    try std.testing.expect(one.len == 1 and std.mem.eql(u8, host.summarize(null, 1, &.{.{ .name = one[0].name }}, false).memory_type, "unified"));
    for ([_][]const u8{ "GPU-private-selector", "-1", "", " " }) |sel| try std.testing.expect(host.nvidiaRows(a, smi, sel).len == 0);
    try std.testing.expect(host.nvidiaRows(a, "/home/example/private-device, 1234, 123\n", null).len == 0);
    const none = host.summarize(null, null, &.{}, false);
    try std.testing.expect(std.mem.eql(u8, none.source, "unavailable") and std.mem.eql(u8, none.memory_type, "unknown"));
}

test "/proc text gives the CPU name and memory" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("AMD EPYC 9004", host.cpuinfo(a, "processor\t: 0\nvendor_id\t: x\nmodel name\t: AMD EPYC 9004\n").?);
    try std.testing.expect(host.cpuinfo(a, "processor\t: 0\nCPU part\t: 0xd4f\n") == null);
    try std.testing.expectEqual(@as(?u64, 131072 * 1024), host.meminfo("MemTotal:       131072 kB\nMemFree: 1 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), host.meminfo("MemTotal: 5 MB\n"));
}

test "a rocm probe runs no nvidia-smi, names no GPU or driver, and passes the save rules" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(host.gpuQuery("rocm") == null and std.mem.eql(u8, host.gpuQuery("cuda").?[0], "nvidia-smi"));
    const probe = host.linux(a, "AMD EPYC 9004", 128 << 30, "", null);
    try std.testing.expect(probe.driver == null and probe.hardware.gpus.len == 0);
    try std.testing.expect(std.mem.eql(u8, probe.hardware.source, "partial") and std.mem.eql(u8, probe.hardware.memory_type, "unknown"));
    const runtime: receipt.Runtime = .{ .tensorfold_version = "0.0.0-test", .backend = "rocm", .python_version = "none", .platform = "linux", .driver = probe.driver };
    const sample: suite.Sample = .{ .fixture_id = "code", .temperature = 0, .repeat = -1, .seed = 1, .status = .@"error", .error_code = .server_error };
    const v = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", .{ .repo_id = "example-org/test-model", .local = false }, runtime, probe.hardware, .{ .tokens = 2, .repetitions = 1, .temperatures = &.{0}, .serial = false, .context_tokens = null, .managed = true }, &.{sample});
    var why: []const u8 = "";
    if (!check.receipt(v, .save, 1767225600000, &why)) {
        std.debug.print("rocm probe failed save: {s}\n", .{why});
        return error.TestUnexpectedResult;
    }
    try std.testing.expect(v.get("runtime").?.get("dependencies").?.object.count() == 0);
}
