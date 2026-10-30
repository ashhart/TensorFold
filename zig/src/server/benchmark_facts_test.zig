//! Checkpoint facts and the hardware parsers on fictional inputs.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model = @import("benchmark_model.zig");
const host = @import("benchmark_host.zig");
const receipt = @import("benchmark_receipt.zig");
const fake = @import("benchmark_fake.zig");
const check = @import("benchmark_check.zig");
const suite = @import("benchmark_suite.zig");
const manifest = @import("benchmark_manifest.zig");
const benchmark = @import("benchmark.zig");

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
    const v = try receipt.value(a, "x", "x", m, .{ .tensorfold_version = "x", .backend = "mlx", .python_version = "none", .platform = "macos" }, .{}, .{ .tokens = 2, .repetitions = 1, .temperatures = &.{0}, .serial = false, .context_tokens = null, .managed = true, .kv_cache = "bf16" }, &.{});
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

test "a server's manifest names the checkpoint a managed run records: the snapshot's revision and the same hashes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try fake.tmpPath(a, &tmp);
    const dir = try checkpoint(a, root, &.{ "models--example--model", "snapshots", revision }, config);
    const managed = try model.describe(a, std.testing.io, "example/model", dir, false, &families, null);
    var env: std.process.Environ.Map = .init(a);
    const doc = try manifest.serveDocument(a, std.testing.io, &env, "9.8.7", "rocm", dir);
    const attached = (try manifest.parse(a, try json.stringify(a, doc, .{}), "rocm")).?;
    try std.testing.expectEqualStrings(managed.revision.?, attached.revision.?);
    try std.testing.expectEqualStrings(managed.config_sha256.?, attached.config_sha256.?);
    try std.testing.expectEqualStrings(managed.tokenizer_sha256.?, attached.tokenizer_sha256.?);
    // the GPU facts a managed run records on this host, cores included, are the ones the server attests
    const probe = host.detect(a, std.testing.io, &env, "rocm");
    try std.testing.expectEqual(probe.hardware.gpus.len, attached.hardware.gpus.len);
    for (probe.hardware.gpus, attached.hardware.gpus) |want, got| {
        try std.testing.expectEqualStrings(want.name, got.name);
        try std.testing.expectEqual(want.memory_bytes, got.memory_bytes);
        try std.testing.expectEqual(want.cores, got.cores);
    }
}

test "a HIP or Metal server's manifest names the receipt's backend, so an attached rocm or mlx run reads it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    for ([_][2][]const u8{ .{ "hip", "rocm" }, .{ "metal", "mlx" }, .{ "cuda", "cuda" } }) |pair| {
        try std.testing.expectEqualStrings(pair[1], manifest.receiptBackend(pair[0]));
        const doc = try manifest.serveDocument(a, std.testing.io, &env, "9.8.7", pair[0], "");
        try std.testing.expectEqualStrings(pair[1], doc.get("runtime").?.get("backend").?.string);
        const facts = (try manifest.parse(a, try json.stringify(a, doc, .{}), pair[1])).?;
        try std.testing.expectEqualStrings(pair[1], facts.runtime.backend);
        // the probe runs for the receipt's backend: on a rocm server it is rocminfo, as a managed rocm run reads it
        const probe = host.detect(a, std.testing.io, &env, pair[1]);
        try std.testing.expectEqual(probe.hardware.gpus.len, facts.hardware.gpus.len);
    }
    try std.testing.expectEqualStrings("", manifest.receiptBackend(""));
}

test "the attested revision: the loaded snapshot's; the environment's only where the directory names none; never a disagreeing one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const other = "fedcba9876543210fedcba9876543210fedcba98";
    const snapshot = "/cache/models--example--model/snapshots/" ++ revision;
    var env: std.process.Environ.Map = .init(a);
    try std.testing.expectEqualStrings(revision, manifest.attestedRevision(&env, snapshot).?);
    try std.testing.expect(manifest.attestedRevision(&env, "/models/private-checkpoint") == null and manifest.attestedRevision(&env, "") == null);
    try std.testing.expect(manifest.attestedRevision(&env, "/cache/snapshots/main") == null);
    try env.put("TF_CHECKPOINT_REVISION", revision);
    try std.testing.expectEqualStrings(revision, manifest.attestedRevision(&env, snapshot).?);
    try env.put("TF_CHECKPOINT_REVISION", other);
    try std.testing.expect(manifest.attestedRevision(&env, snapshot) == null);
    try std.testing.expectEqualStrings(other, manifest.attestedRevision(&env, "/models/private-checkpoint").?);
    try std.testing.expectEqualStrings(other, manifest.attestedRevision(&env, "").?);
    try std.testing.expectEqualStrings(revision, model.snapshotRevision(snapshot).?);
    for ([_][]const u8{ "/cache/snapshots/" ++ "0123456789ABCDEF0123456789abcdef01234567", "/cache/snapshots/0123", "/cache/other/" ++ revision, revision }) |no|
        try std.testing.expect(model.snapshotRevision(no) == null);
}

test "a manifest's GPU cores reach the attached receipt; cores past the board's range refuse the manifest" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"schema_version":1,"runtime":{"tensorfold_version":"1.1.1","backend":"rocm","python_version":"none","platform":"linux"},"model":{},"hardware":{"cpu_model":"AMD RYZEN AI MAX+ 395 with Radeon 8060S","system_memory_bytes":134217728000,"memory_type":"unified","gpus":[{"name":"AMD Radeon 8060S Graphics","memory_bytes":128849018880,"cores":40}],"gpu_count":1,"source":"detected"}}
    ;
    const facts = (try manifest.parse(a, body, "rocm")).?;
    try std.testing.expectEqual(@as(?u64, 40), facts.hardware.gpus[0].cores);
    const too_many = try std.mem.replaceOwned(u8, a, body, "\"cores\":40", "\"cores\":100001");
    try std.testing.expect((try manifest.parse(a, too_many, "rocm")) == null);
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

test "the tokenizer hash covers the chat template and not the weights; malformed config is refused" {
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
    try std.testing.expectError(error.InvalidConfig, model.describe(a, io, dir, dir, true, &families, null));
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
    for ([_][]const u8{ "private-machine", "/home/example/private", "NVIDIA /home/example/private", "NVIDIA NPU", "AMD w/home/example", "AMD Ryzen w/ /home", long, "Apple M5 Max Extra", "Apple M", "Intel user@example.test" }) |bad| try std.testing.expect(host.modelName(a, bad) == null);
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
    const spark = host.nvidiaRows(a, "NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU), 64000, 617.42\nNVIDIA NPU, [N/A], 617.42\n", null);
    try std.testing.expect(spark.len == 1 and spark[0].memory_bytes.? == 64000 << 20);
    try std.testing.expectEqualStrings("unified", host.summarize("ARMv8", gib128, &.{.{ .name = spark[0].name, .memory_bytes = spark[0].memory_bytes }}, false).memory_type);
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

const rocminfo_strix =
    \\ROCk module is loaded
    \\HSA Agents
    \\*******
    \\Agent 1
    \\*******
    \\  Name:                    AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
    \\  Uuid:                    CPU-XX
    \\  Marketing Name:          AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
    \\  Vendor Name:             CPU
    \\  Device Type:             CPU
    \\  Compute Unit:            32
    \\  Pool Info:
    \\    Pool 1
    \\      Segment:                 GLOBAL; FLAGS: FINE GRAINED
    \\      Size:                    127838212(0x79ea604) KB
    \\*******
    \\Agent 2
    \\*******
    \\  Name:                    gfx1151
    \\  Uuid:                    GPU-XX
    \\  Marketing Name:          Radeon 8060S Graphics
    \\  Vendor Name:             AMD
    \\  Device Type:             GPU
    \\  Chip ID:                 5510(0x1586)
    \\  Compute Unit:            40
    \\  Pool Info:
    \\    Pool 1
    \\      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
    \\      Size:                    125829120(0x7800000) KB
    \\    Pool 2
    \\      Segment:                 GROUP
    \\      Size:                    64(0x40) KB
    \\  ISA Info:
    \\    ISA 1
    \\      Name:                    amdgcn-amd-amdhsa--gfx1151
    \\
;

const rocminfo_two_dgpus =
    \\*******
    \\Agent 2
    \\*******
    \\  Marketing Name:          AMD Radeon PRO W7800
    \\  Vendor Name:             AMD
    \\  Device Type:             GPU
    \\  Compute Unit:            70
    \\      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
    \\      Size:                    50331648(0x3000000) KB
    \\*******
    \\Agent 3
    \\*******
    \\  Marketing Name:          AMD Radeon PRO V620
    \\  Vendor Name:             AMD
    \\  Device Type:             GPU
    \\  Compute Unit:            72
    \\      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
    \\      Size:                    31440896(0x1dfc000) KB
    \\
;

test "a rocm probe names rocminfo's GPU agents, runs no nvidia-smi, reports no driver, and passes the save rules" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(std.mem.eql(u8, host.gpuQuery("rocm").?[0], "rocminfo") and std.mem.eql(u8, host.gpuQuery("cuda").?[0], "nvidia-smi"));
    const cpu = host.cpuinfo(a, "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n");
    try std.testing.expectEqualStrings("AMD RYZEN AI MAX+ 395 with Radeon 8060S", cpu.?);
    try std.testing.expect(check.name(cpu.?));
    const probe = host.rocm(a, cpu, 128 << 30, rocminfo_strix, null);
    try std.testing.expect(probe.driver == null and probe.hardware.gpus.len == 1);
    const g = probe.hardware.gpus[0];
    try std.testing.expectEqualStrings("AMD Radeon 8060S Graphics", g.name);
    try std.testing.expectEqual(@as(?u64, 125829120 * 1024), g.memory_bytes);
    try std.testing.expectEqual(@as(?u64, 40), g.cores);
    try std.testing.expect(std.mem.eql(u8, probe.hardware.source, "detected") and std.mem.eql(u8, probe.hardware.memory_type, "unified"));
    try std.testing.expectEqualStrings("partial", host.rocm(a, null, 128 << 30, rocminfo_strix, null).hardware.source);
    const runtime: receipt.Runtime = .{ .tensorfold_version = "0.0.0-test", .backend = "rocm", .python_version = "none", .platform = "linux", .driver = probe.driver };
    const sample: suite.Sample = .{ .fixture_id = "code", .temperature = 0, .repeat = -1, .seed = 1, .status = .@"error", .error_code = .server_error };
    const v = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", .{ .repo_id = "example-org/test-model", .local = false }, runtime, probe.hardware, .{ .tokens = 2, .repetitions = 1, .temperatures = &.{0}, .serial = false, .context_tokens = null, .managed = true, .kv_cache = "bf16" }, &.{sample});
    var why: []const u8 = "";
    if (!check.receipt(v, .save, 1767225600000, &why)) {
        std.debug.print("rocm probe failed save: {s}\n", .{why});
        return error.TestUnexpectedResult;
    }
    try std.testing.expect(v.get("runtime").?.get("dependencies").?.object.count() == 0);
    // what the board validates in the CPU, GPU and checkpoint fields passes its publish rules; only the backend waits
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try checkpoint(a, try fake.tmpPath(a, &tmp), &.{ "snapshots", revision }, config);
    const checkpoint_facts = try model.describe(a, std.testing.io, "example-org/test-model", dir, false, &families, null);
    try std.testing.expect(checkpoint_facts.revision != null and checkpoint_facts.config_sha256 != null and checkpoint_facts.tokenizer_sha256 != null);
    const settings: receipt.Settings = .{ .tokens = 2, .repetitions = 1, .temperatures = &.{0}, .serial = false, .context_tokens = null, .managed = true, .rank_count = 1, .kv_cache = "bf16" };
    const rocm_receipt = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", checkpoint_facts, runtime, probe.hardware, settings, &.{sample});
    try std.testing.expect(!check.receipt(rocm_receipt, .publish, 1767225600000, &why));
    try std.testing.expectEqualStrings(check.rocm_local, why);
    var board_backend = runtime;
    board_backend.backend = "cuda";
    const same_facts = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", checkpoint_facts, board_backend, probe.hardware, settings, &.{sample});
    if (!check.receipt(same_facts, .publish, 1767225600000, &why)) {
        std.debug.print("rocm identity failed the publish rules: {s}\n", .{why});
        return error.TestUnexpectedResult;
    }
    // the client says what the board will not rank: the facts a probe or checkpoint left out
    try std.testing.expect(try benchmark.unranked(a, same_facts) == null);
    const no_gpu = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", checkpoint_facts, runtime, host.rocm(a, cpu, 128 << 30, "", null).hardware, settings, &.{sample});
    try std.testing.expectEqualStrings("GPU", (try benchmark.unranked(a, no_gpu)).?);
    var unpinned = checkpoint_facts;
    unpinned.revision = null;
    unpinned.config_sha256 = null;
    const loose = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", unpinned, runtime, probe.hardware, settings, &.{sample});
    try std.testing.expectEqualStrings("checkpoint revision, configuration digest", (try benchmark.unranked(a, loose)).?);
    const local = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", .{ .local = true }, runtime, .{}, settings, &.{sample});
    try std.testing.expect(try benchmark.unranked(a, local) == null);
}

test "rocm GPUs: discrete cards are dedicated, visible-device indices select, and missing rocminfo leaves the GPUs unknown" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const both = host.rocm(a, "AMD EPYC 9004", 128 << 30, rocminfo_two_dgpus, null);
    try std.testing.expect(both.hardware.gpus.len == 2 and std.mem.eql(u8, both.hardware.memory_type, "dedicated"));
    try std.testing.expectEqualStrings("AMD Radeon PRO V620", both.hardware.gpus[1].name);
    const second = host.rocm(a, "AMD EPYC 9004", 128 << 30, rocminfo_two_dgpus, "1");
    try std.testing.expect(second.hardware.gpus.len == 1 and second.hardware.gpus[0].cores.? == 72);
    for ([_][]const u8{ "-1", " ", "x" }) |sel| try std.testing.expect(host.rocm(a, null, null, rocminfo_two_dgpus, sel).hardware.gpus.len == 0);
    const none = host.rocm(a, "AMD EPYC 9004", 128 << 30, "", null);
    try std.testing.expect(none.hardware.gpus.len == 0 and std.mem.eql(u8, none.hardware.source, "partial") and std.mem.eql(u8, none.hardware.memory_type, "unknown"));
}

// the first GPU agent's name fails the allowlist; it still holds index 0 of the visible-device indices
const rocminfo_unnamed_first =
    \\*******
    \\Agent 2
    \\*******
    \\  Marketing Name:          /home/example/private
    \\  Vendor Name:             AMD
    \\  Device Type:             GPU
    \\  Compute Unit:            70
    \\      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
    \\      Size:                    50331648(0x3000000) KB
    \\*******
    \\Agent 3
    \\*******
    \\  Marketing Name:          AMD Radeon PRO V620
    \\  Vendor Name:             AMD
    \\  Device Type:             GPU
    \\  Compute Unit:            72
    \\      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
    \\      Size:                    31440896(0x1dfc000) KB
    \\
;

test "rocm visible-device indices count every GPU agent: selecting an unreportable agent lists no GPU, never the next" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v620: []const []const u8 = &.{"AMD Radeon PRO V620"};
    // with no list or 0,1 HIP sees both agents and the probe names one, so the list is short and the source partial
    const cases = [_]struct { key: []const u8, visible: []const u8, want: []const []const u8, source: []const u8 }{
        .{ .key = "", .visible = "", .want = v620, .source = "partial" },
        .{ .key = "HIP_VISIBLE_DEVICES", .visible = "1", .want = v620, .source = "detected" },
        .{ .key = "HIP_VISIBLE_DEVICES", .visible = "0", .want = &.{}, .source = "partial" },
        .{ .key = "HIP_VISIBLE_DEVICES", .visible = "0,1", .want = v620, .source = "partial" },
        .{ .key = "CUDA_VISIBLE_DEVICES", .visible = "1", .want = v620, .source = "detected" },
        .{ .key = "CUDA_VISIBLE_DEVICES", .visible = "0", .want = &.{}, .source = "partial" },
    };
    const runtime: receipt.Runtime = .{ .tensorfold_version = "0.0.0-test", .backend = "rocm", .python_version = "none", .platform = "linux" };
    const settings: receipt.Settings = .{ .tokens = 2, .repetitions = 1, .temperatures = &.{0}, .serial = false, .context_tokens = null, .managed = true, .kv_cache = "bf16" };
    const sample: suite.Sample = .{ .fixture_id = "code", .temperature = 0, .repeat = -1, .seed = 1, .status = .@"error", .error_code = .server_error };
    var wrong: usize = 0;
    for (cases) |c| {
        var env: std.process.Environ.Map = .init(a);
        if (c.key.len > 0) try env.put(c.key, c.visible);
        const hw = host.rocm(a, "AMD EPYC 9004", 128 << 30, rocminfo_unnamed_first, host.rocmVisible(&env)).hardware;
        const names = try a.alloc([]const u8, hw.gpus.len);
        for (hw.gpus, names) |g, *n| n.* = g.name;
        const got = try std.fmt.allocPrint(a, "[{s}] {s}", .{ try std.mem.join(a, ", ", names), hw.source });
        const want = try std.fmt.allocPrint(a, "[{s}] {s}", .{ try std.mem.join(a, ", ", c.want), c.source });
        if (!std.mem.eql(u8, got, want)) {
            std.debug.print("{s}={s}: listed GPUs {s}, want {s}\n", .{ c.key, c.visible, got, want });
            wrong += 1;
            continue;
        }
        if (hw.gpus.len == 1) try std.testing.expectEqual(@as(?u64, 72), hw.gpus[0].cores);
        // the receipt's gpu_count is the length of that list, and the save rules accept it
        const v = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", .{ .repo_id = "example-org/test-model", .local = false }, runtime, hw, settings, &.{sample});
        const count = v.get("hardware").?.get("gpu_count").?;
        try std.testing.expect(if (hw.gpus.len == 0) count == .null else count.int64().? == @as(i64, @intCast(hw.gpus.len)));
        var why: []const u8 = "";
        if (!check.receipt(v, .save, 1767225600000, &why)) {
            std.debug.print("{s}={s}: save refused: {s}\n", .{ c.key, c.visible, why });
            return error.TestUnexpectedResult;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), wrong);
    // HIP reads HIP_VISIBLE_DEVICES, else CUDA_VISIBLE_DEVICES; ROCr applies ROCR_VISIBLE_DEVICES inside rocminfo
    var env: std.process.Environ.Map = .init(a);
    try env.put("ROCR_VISIBLE_DEVICES", "0");
    try std.testing.expect(host.rocmVisible(&env) == null);
    try env.put("CUDA_VISIBLE_DEVICES", "0");
    try env.put("HIP_VISIBLE_DEVICES", "1");
    try std.testing.expectEqualStrings("1", host.rocmVisible(&env).?);
}
