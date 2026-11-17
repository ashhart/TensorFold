//! CPU checks for halt ordering, with the real lane host and fake target.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    b.step("test-lone-copy-host", "Lone-copy pricing and all existing lane depth/verification regressions without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes })).step);
    b.step("test-metal-cost-host", "Metal window timing and retained-price healing without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"Metal window prices:"} })).step);
    b.step("test-metal-cost-review-host", "Warm-budget and tile-edge regression controls without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"Metal review:"} })).step);
    @import("metal_cost_host.zig").steps(b, lanes);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const interrupted = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_host_interrupted_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    b.step("test-lane-interrupted-host", "Interrupted prompt retention and clean retries without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = interrupted, .filters = &.{"interrupted prefill:"} })).step);
    const seeds = b.addOptions(); // -Dexplore-seeds=N sweeps N schedules; -Dexplore-seed=S replays one
    seeds.addOption(u32, "seeds", b.option(u32, "explore-seeds", "Schedules test-lane-explore-host runs a scenario") orelse 200);
    seeds.addOption(?u64, "seed", b.option(u64, "explore-seed", "One lane host schedule to replay (a failure prints it)"));
    seeds.addOption(bool, "tally", b.option(bool, "explore-tally", "Print what the explored schedules did") orelse false);
    const explore = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_host_explore_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    explore.addOptions("explore_options", seeds);
    b.step("test-lane-explore-host", "Lane host invariants under seeded arrivals, cancels, room and pressure, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = explore, .filters = &.{"explore:"} })).step);
    b.step("test-lanes-sampler-launch-host", "A Metal sampler launch of many rows binds each row's one-row bytes, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"sampler launch:"} })).step);
    b.step("test-lane-invariance-host", "Batch-composition token, stop and RNG-position invariance with a fake target").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"lane invariance:"} })).step);
    const safe_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .safe, .link_libc = true });
    const strand = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_host_strand_test.zig"), .target = b.graph.host, .optimize = .safe, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = safe_lanes }} });
    b.step("test-lane-strand-host", "Refused, cancelled and failed requests give back every lane, page and slot without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = strand, .filters = &.{"lane strand:"} })).step);
    b.step("test-lane-condition-stress-host", "A held score asker's wake is not taken by the lane thread's timed idle waits, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"lane condition stress:"} })).step);
    const halt = b.addTest(.{ .root_module = api, .filters = &.{ "halt (", "halt fails" } });
    b.step("test-lane-halt-host", "Halt ordering with a fake backend and no GPU dependencies").dependOn(&b.addRunArtifact(halt).step);
    b.step("test-lane-lone-host", "Existing lone-driver handoff and reclaim with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"lone driver"} })).step);
    b.step("test-lane-api-host", "Shared lane host and engine API CPU regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    b.step("test-shared-price-learning-host", "Independent plain prices and signed row learning together without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{ "shared combined prices:", "shared price modes:" } })).step);
    b.step("test-hold-pass-host", "A drafter's timed hold pass priced against copy rounds under a legacy cap, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"hold pass:"} })).step);
    b.step("test-wait-reasons-host", "Stored wait reasons in the load ledger with a fake backend and planner").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"wait reason ledger:"} })).step);
    const spares = b.step("test-cache-spares-host", "Fallback and guarded keeps preserve cache retention");
    for ([_][]const u8{ "lane_cache_promise_test.zig", "prompt_cache_promised_test.zig" }) |file| {
        const m = b.createModule(.{ .root_source_file = b.path(b.fmt("zig/src/core/{s}", .{file})), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
        spares.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"queued cache: spares:"} })).step);
    }
    b.step("test-conversation-cache-host", "Returning conversation eviction and queued-resume regressions without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{ "newest cache:", "queued cache:" } })).step);
    b.step("test-cache-progress-host", "One-state follow-ups and new conversations keep their next endpoint").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"queued cache: progress"} })).step);
    b.step("test-snapshot-index-host", "Prompt-cache lookup index against the whole-store scan, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"snapshot index:"} })).step);
    b.step("test-rows-prefetch-host", "Disk states read ahead for queued requests: equality, cancellation, fallbacks").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"prefetch:"} })).step);
    b.step("test-embedding-lane-host", "Prompt-only embedding jobs on the actual lane worker without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"embedding lane:"} })).step);
    b.step("test-prefill-service-host", "The opt-in prompt-row budget beside decoding streams on the lane host with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"prefill service:"} })).step);
    const copies = b.step("test-root-copy-host", "Root copy precedence, drafter absorption and real worker checks without GPU");
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"root copy:"} })).step);
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"root copy:"} })).step);
}
