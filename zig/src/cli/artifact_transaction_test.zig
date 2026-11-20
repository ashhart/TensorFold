const std = @import("std");
const transaction = @import("artifact_transaction.zig");
const resolution = @import("core").artifact_resolution;
const release = @import("core").drafter_release;
const sys = @import("sys");
const catalog = @import("catalog.zig");

const rev0 = "0123456789abcdef0123456789abcdef01234567";
const rev1 = "1123456789abcdef0123456789abcdef01234567";
const rev2 = "2123456789abcdef0123456789abcdef01234567";
const rev3 = "3123456789abcdef0123456789abcdef01234567";
const rev4 = "4123456789abcdef0123456789abcdef01234567";

fn artifact(repo: []const u8, revision: []const u8, byte: u8) resolution.Artifact {
    return .{
        .repository = repo,
        .revision = revision,
        .family = "test-target",
        .tensor_format = "test-q4",
        .template = "chat-v1",
        .vocabulary = "tokens-v1",
        .checkpoint = @splat(byte),
        .context = 65536,
    };
}

fn request() resolution.Request {
    var draft = artifact("Org/Drafter", rev2, 2);
    draft.family = "test-drafter";
    draft.context = 32768;
    return .{
        .base = artifact("Org/Target", rev1, 1),
        .drafter = draft,
        .capability = .{ .backend = "metal", .device = "M4-Max" },
        .context = 49152,
    };
}

fn plan(req: resolution.Request) !transaction.Plan {
    return switch (transaction.resolve(req)) {
        .plan => |p| p,
        .refusal => |r| {
            std.debug.print("unexpected refusal {s}: {s}\n", .{ r.name(), r.message() });
            return error.UnexpectedRefusal;
        },
    };
}

fn expectRefusal(req: resolution.Request, kind: transaction.Refusal.Kind) !void {
    switch (transaction.resolve(req)) {
        .plan => return error.ExpectedRefusal,
        .refusal => |r| {
            try std.testing.expectEqual(kind, r.kind);
            try std.testing.expect(r.message().len != 0);
        },
    }
}

fn expectCheckedRefusal(req: resolution.Request, constraints: transaction.Constraints, kind: transaction.Refusal.Kind) !void {
    switch (transaction.resolveChecked(req, constraints)) {
        .plan => return error.ExpectedRefusal,
        .refusal => |r| try std.testing.expectEqual(kind, r.kind),
    }
}

fn seedRepo(a: std.mem.Allocator, io: std.Io, root: []const u8, repo: []const u8, revision: []const u8) !void {
    const dir_name = switch (std.mem.eql(u8, repo, "Org/Target")) {
        true => "models--Org--Target",
        false => "models--Org--Drafter",
    };
    const snapshot = try std.fs.path.join(a, &.{ root, dir_name, "snapshots", revision });
    const refs = try std.fs.path.join(a, &.{ root, dir_name, "refs" });
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, snapshot);
    try cwd.createDirPath(io, refs);
    try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snapshot, "config.json" }), .data = "{\"model_type\":\"qwen4_exp\"}" });
    try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snapshot, "weights.safetensors" }), .data = "old weights" });
    try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ refs, "main" }), .data = revision });
}

fn writeHead(a: std.mem.Allocator, io: std.Io, root: []const u8, folder: []const u8, revision: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, folder, "refs", "main" }), .data = revision });
}

fn readHead(a: std.mem.Allocator, io: std.Io, root: []const u8, folder: []const u8) ![]const u8 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, folder, "refs", "main" }), a, .limited(128));
    return std.mem.trim(u8, text, " \r\n");
}

fn stageConfig(a: std.mem.Allocator, io: std.Io, directory: []const u8, body: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ directory, "config.json" }), .data = body });
}

fn cacheRoot(a: std.mem.Allocator, tmp: std.testing.TmpDir) ![]const u8 {
    return std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "hub" });
}

fn kolibri() *const catalog.Entry {
    for (&catalog.entries) |*entry| if (std.mem.eql(u8, entry.repo, "Aleph-Alpha/Kolibri-1")) return entry;
    unreachable;
}

const drafter_json =
    \\{"format":"tensorfold-drafter","version":1,"architecture":"block-drafter",
    \\ "target":{"repo":"Org/Target","revision":"1123456789abcdef0123456789abcdef01234567","layers":2,"hidden":4,"vocab":8,"embed_scale":1.0,"tied":false,"softcap":null,"head_bias":false},
    \\ "taps":{"ids":[2],"stream":"residual","normed":true,"dtype":"bf16","row_bytes":8},
    \\ "drafter":{"hidden":4,"heads":1,"head_dim":2,"ffn":8,"layers":1,"block":2,"window":16,"eps":0.0001,"theta":10000.0,"pred_rank":0,"spine_rank":0},
    \\ "draft_vocab":{"file":"draft_vocab.json","size":2},"weights":"model.safetensors",
    \\ "serving":{"depth":1,"body":"bf16","head":"bf16"}}
;

fn adapter() release.Adapter {
    return .{ .layers = 2, .hidden = 4, .vocab = 8, .embed_scale = 1.0, .tied = false, .softcap = null, .head_bias = false };
}

test "one resolver produces an exact plan identity or a named refusal" {
    const req = request();
    const first = try plan(req);
    const again = try plan(req);
    try std.testing.expectEqual(first.resolved.id, again.resolved.id);
    try std.testing.expectEqual(@as(u32, 32768), first.resolved.effective_context);

    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try first.describe(&out.writer);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "transaction") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), rev1) != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), rev2) != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "M4-Max") != null);

    var changed = req;
    changed.capability.backend = "cuda";
    const other = try plan(changed);
    try std.testing.expect(!std.mem.eql(u8, &first.resolved.id, &other.resolved.id));
    changed = req;
    changed.drafter.?.checkpoint[0] +%= 1;
    const changed_draft = try plan(changed);
    try std.testing.expect(!std.mem.eql(u8, &first.resolved.id, &changed_draft.resolved.id));

    changed = req;
    changed.base.revision = "main";
    try expectRefusal(changed, .unpinned_revision);
    changed = req;
    changed.drafter.?.template = "other-chat";
    try expectRefusal(changed, .template_mismatch);
    changed = req;
    changed.capability.device = "";
    try expectRefusal(changed, .capability_missing);
    changed = req;
    changed.drafter.?.repository = changed.base.repository;
    try expectRefusal(changed, .artifact_collision);
}

test "release compatibility and machine capability are admitted before a plan exists" {
    var why: release.Why = .{};
    var manifest = try release.Manifest.parse(std.testing.allocator, drafter_json, &why);
    defer manifest.deinit();
    const req = request();
    const resolved = transaction.resolveChecked(req, .{ .drafter_release = &manifest.config, .target_adapter = adapter() });
    try std.testing.expect(resolved == .plan);
    manifest.config.version = 2;
    try expectCheckedRefusal(req, .{ .drafter_release = &manifest.config, .target_adapter = adapter() }, .bad_drafter_release);
    manifest.config.version = 1;

    var bad_adapter = adapter();
    bad_adapter.layers = 3;
    try expectCheckedRefusal(req, .{ .drafter_release = &manifest.config, .target_adapter = bad_adapter }, .adapter_mismatch);
    try expectCheckedRefusal(req, .{ .drafter_release = &manifest.config }, .target_adapter_missing);
    var wrong_target = req;
    wrong_target.base.revision = rev3;
    try expectCheckedRefusal(wrong_target, .{ .drafter_release = &manifest.config, .target_adapter = adapter() }, .drafter_target_mismatch);

    const entry = kolibri();
    var cuda_req = request();
    cuda_req.base.repository = entry.repo;
    cuda_req.base.revision = entry.revision.?;
    cuda_req.drafter = null;
    cuda_req.capability = .{ .backend = "cuda", .device = "GB10" };
    var metal: sys.machine.Machine = .{ .backend = .metal, .memory = 256 * sys.machine.gib, .gpu_limit = 256 * sys.machine.gib };
    metal.setChip("M5 Ultra");
    try expectCheckedRefusal(cuda_req, .{ .catalog_entry = entry, .machine = &metal }, .backend_unsupported);
    var small_cuda: sys.machine.Machine = .{ .backend = .cuda, .memory = 1, .gpu_limit = 1 };
    small_cuda.setChip("GB10");
    try expectCheckedRefusal(cuda_req, .{ .catalog_entry = entry, .machine = &small_cuda }, .capacity_exceeded);
}

test "both staged snapshots replace both heads only after the immutable tuple is revalidated" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try cacheRoot(a, tmp);
    try seedRepo(a, io, root, "Org/Target", rev0);
    try seedRepo(a, io, root, "Org/Drafter", rev1);

    const req = request();
    const p = try plan(req);
    var tx = try p.begin(a, io, root);
    var open = true;
    defer if (open) tx.deinit();
    const prepared = try std.Io.Dir.cwd().readFileAlloc(io, tx.receipt_path, a, .limited(1 << 20));
    try std.testing.expect(std.mem.indexOf(u8, prepared, "\"state\":\"prepared\"") != null);
    try std.testing.expectEqualStrings(rev0, try readHead(a, io, root, "models--Org--Target"));
    try std.testing.expectEqualStrings(rev1, try readHead(a, io, root, "models--Org--Drafter"));
    try std.testing.expect((try transaction.exactCachedSnapshot(a, io, root, artifact("Org/Target", rev0, 1))) != null);
    var symbolic = artifact("Org/Target", "main", 1);
    symbolic.revision = "main";
    try std.testing.expect((try transaction.exactCachedSnapshot(a, io, root, symbolic)) == null);

    try stageConfig(a, io, tx.targetStaging(), "{\"model_type\":\"qwen4_exp\",\"new\":true}");
    try stageConfig(a, io, tx.drafterStaging().?, drafter_json);
    try tx.commit(req);
    try std.testing.expectEqual(transaction.State.committed, tx.state);
    try std.testing.expectEqualStrings(rev1, try readHead(a, io, root, "models--Org--Target"));
    try std.testing.expectEqualStrings(rev2, try readHead(a, io, root, "models--Org--Drafter"));
    const target_config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "models--Org--Target/snapshots", rev1, "config.json" }), a, .limited(1 << 20));
    try std.testing.expect(std.mem.indexOf(u8, target_config, "\"new\":true") != null);
    tx.deinit();
    open = false;
    try std.testing.expectEqual(transaction.Recovery.committed, try transaction.recover(a, io, root, p.resolved.id));
}

test "an existing snapshot must match every staged byte before either visible head changes" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try cacheRoot(a, tmp);
    try seedRepo(a, io, root, "Org/Target", rev0);
    try seedRepo(a, io, root, "Org/Target", rev1);
    try writeHead(a, io, root, "models--Org--Target", rev0);
    var req = request();
    req.drafter = null;
    var tx = try (try plan(req)).begin(a, io, root);
    defer tx.deinit();
    try stageConfig(a, io, tx.targetStaging(), "{\"model_type\":\"qwen4_exp\"}");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ tx.targetStaging(), "weights.safetensors" }), .data = "new weights" });
    try std.testing.expectError(error.CachedSnapshotMismatch, tx.commit(req));
    try std.testing.expectEqualStrings(rev0, try readHead(a, io, root, "models--Org--Target"));
    const cached = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "models--Org--Target/snapshots", rev1, "weights.safetensors" }), a, .limited(1 << 20));
    try std.testing.expectEqualStrings("old weights", cached);
    try tx.rollback();
    try std.testing.expectEqualStrings(rev0, try readHead(a, io, root, "models--Org--Target"));
}

test "a byte-identical cached snapshot is reused and the staged copy is discarded" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try cacheRoot(a, tmp);
    try seedRepo(a, io, root, "Org/Target", rev0);
    try seedRepo(a, io, root, "Org/Target", rev1);
    try writeHead(a, io, root, "models--Org--Target", rev0);
    const cached_weights = try std.fs.path.join(a, &.{ root, "models--Org--Target/snapshots", rev1, "weights.safetensors" });
    try std.Io.Dir.cwd().deleteFile(io, cached_weights);
    try std.Io.Dir.cwd().symLink(io, "../../blobs/abc", cached_weights, .{});
    var req = request();
    req.drafter = null;
    var tx = try (try plan(req)).begin(a, io, root);
    defer tx.deinit();
    try stageConfig(a, io, tx.targetStaging(), "{\"model_type\":\"qwen4_exp\"}");
    try std.Io.Dir.cwd().symLink(io, "../../blobs/abc", try std.fs.path.join(a, &.{ tx.targetStaging(), "weights.safetensors" }), .{});
    try tx.commit(req);
    try std.testing.expectEqual(transaction.State.committed, tx.state);
    try std.testing.expectEqualStrings(rev1, try readHead(a, io, root, "models--Org--Target"));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, tx.work_root, .{}));
}

test "wrong head or a changed tuple leaves every visible head untouched and rollback discards staging" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try cacheRoot(a, tmp);
    try seedRepo(a, io, root, "Org/Target", rev0);
    try seedRepo(a, io, root, "Org/Drafter", rev1);

    const req = request();
    const p = try plan(req);
    var tx = try p.begin(a, io, root);
    defer tx.deinit();
    try stageConfig(a, io, tx.targetStaging(), "{}");
    try stageConfig(a, io, tx.drafterStaging().?, drafter_json);
    var changed = req;
    changed.drafter.?.checkpoint[31] +%= 1;
    try std.testing.expectError(error.ArtifactTupleChanged, tx.commit(changed));
    try writeHead(a, io, root, "models--Org--Target", rev4);
    try std.testing.expectError(error.WrongHead, tx.commit(req));
    try tx.rollback();
    try std.testing.expectEqualStrings(rev4, try readHead(a, io, root, "models--Org--Target"));
    try std.testing.expectEqualStrings(rev1, try readHead(a, io, root, "models--Org--Drafter"));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, tx.work_root, .{}));
    const committed = try std.Io.Dir.cwd().readFileAlloc(io, tx.receipt_path, a, .limited(1 << 20));
    try std.testing.expect(std.mem.indexOf(u8, committed, "\"state\":\"rolled_back\"") != null);
}

test "a restarted publishing transaction rolls both artifacts back and never resurrects a missing head" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try cacheRoot(a, tmp);
    try seedRepo(a, io, root, "Org/Target", rev0);

    const req = request();
    const p = try plan(req);
    var tx = try p.begin(a, io, root);
    const receipt_path = tx.receipt_path;
    var open = true;
    defer if (open) tx.deinit();
    try stageConfig(a, io, tx.targetStaging(), "{}");
    try stageConfig(a, io, tx.drafterStaging().?, drafter_json);
    const new_target = try std.fs.path.join(a, &.{ root, "models--Org--Target/snapshots", rev1 });
    try std.Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(new_target).?);
    try std.Io.Dir.cwd().rename(tx.targetStaging(), std.Io.Dir.cwd(), new_target, io);
    try writeHead(a, io, root, "models--Org--Target", rev1);
    const receipt = try std.Io.Dir.cwd().readFileAlloc(io, receipt_path, a, .limited(1 << 20));
    const publishing = try std.mem.replaceOwned(u8, a, receipt, "\"state\":\"prepared\"", "\"state\":\"publishing\"");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = receipt_path, .data = publishing });
    tx.deinit();
    open = false;

    try std.testing.expectEqual(transaction.Recovery.rolled_back, try transaction.recover(a, io, root, p.resolved.id));
    try std.testing.expectEqualStrings(rev0, try readHead(a, io, root, "models--Org--Target"));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, new_target, .{}));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Drafter/refs/main" }), .{}));
    const rolled_back = try std.Io.Dir.cwd().readFileAlloc(io, receipt_path, a, .limited(1 << 20));
    try std.testing.expect(std.mem.indexOf(u8, rolled_back, "\"state\":\"rolled_back\"") != null);
    try std.testing.expectEqual(transaction.Recovery.already_rolled_back, try transaction.recover(a, io, root, p.resolved.id));
}

test "restart refuses a receipt that no longer describes its immutable pinned tuple" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try cacheRoot(a, tmp);
    const p = try plan(request());
    var tx = try p.begin(a, io, root);
    const receipt_path = tx.receipt_path;
    const receipt = try std.Io.Dir.cwd().readFileAlloc(io, receipt_path, a, .limited(1 << 20));
    const committed = try std.mem.replaceOwned(u8, a, receipt, "\"state\":\"prepared\"", "\"state\":\"committed\"");
    const at = std.mem.indexOf(u8, committed, rev1).?;
    const damaged = try a.dupe(u8, committed);
    @memset(damaged[at .. at + rev1.len], 'z');
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = receipt_path, .data = damaged });
    tx.deinit();
    try std.testing.expectError(error.BadTransactionReceipt, transaction.recover(a, io, root, p.resolved.id));
}

test "cached snapshot equality compares relative symlink target bytes" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "staged", "cached" }) |root| {
        const snapshot = try std.fs.path.join(a, &.{ root, "snapshot" });
        defer a.free(snapshot);
        try tmp.dir.createDirPath(io, snapshot);
        const blob_dir = try std.fs.path.join(a, &.{ root, "blobs" });
        defer a.free(blob_dir);
        try tmp.dir.createDirPath(io, blob_dir);
        const blob = try std.fs.path.join(a, &.{ root, "blobs/weight" });
        defer a.free(blob);
        const config = try std.fs.path.join(a, &.{ root, "snapshot/config.json" });
        defer a.free(config);
        try tmp.dir.writeFile(io, .{ .sub_path = config, .data = "{}" });
        const link = try std.fs.path.join(a, &.{ root, "snapshot/weight" });
        defer a.free(link);
        try tmp.dir.symLink(io, "../blobs/weight", link, .{});
        try tmp.dir.writeFile(io, .{ .sub_path = blob, .data = if (std.mem.eql(u8, root, "staged")) "staged" else "cached" });
    }
    const staged = try tmp.dir.realPathFileAlloc(io, "staged/snapshot", a);
    defer a.free(staged);
    const cached = try tmp.dir.realPathFileAlloc(io, "cached/snapshot", a);
    defer a.free(cached);
    try std.testing.expect(!try @import("snapshot_equal.zig").equal(io, staged, cached));
}
