//! DeepSeek-V4.1 drafting: DSpark glue, cost-derived depth, trees and lanes over iface.zig's `Target` / `Pass`.

pub const costs = @import("costs.zig");
pub const Costs = costs.Costs;
pub const calib = @import("calib.zig");
pub const joint = @import("joint.zig");
pub const depth = @import("depth.zig");
pub const Depth = depth.Depth;
pub const tree = @import("tree.zig");
pub const dspark = @import("dspark.zig");
pub const iface = @import("iface.zig");
pub const policy = @import("policy.zig");
pub const lanes = @import("lanes.zig");
pub const Lanes = lanes.Lanes;
pub const twin = @import("twin.zig");
pub const drive = @import("drive.zig");
pub const branches = @import("branches.zig");
pub const lookup = @import("lookup.zig");
pub const calib_env = @import("calib_env.zig");
pub const calib_measure = @import("calib_measure.zig");
pub const branches_oracle = @import("branches_oracle.zig");
pub const spec = @import("spec.zig");

test {
    _ = @import("ownership_test.zig");
    _ = costs;
    _ = calib;
    _ = joint;
    _ = depth;
    _ = tree;
    _ = dspark;
    _ = @import("dspark_test.zig"); // the chain lanes, bit for bit
    _ = iface;
    _ = policy;
    _ = lanes;
    _ = twin;
    _ = drive;
    _ = branches;
    _ = calib_env;
    _ = @import("calib_env_test.zig");
    _ = calib_measure;
    _ = @import("calib_measure_test.zig");
    _ = branches_oracle;
    _ = @import("branches_test.zig");
    _ = @import("drive_test.zig");
    _ = @import("lanes_test.zig");
    _ = @import("golden_test.zig");
    _ = spec;
    _ = @import("spec_test.zig");
    _ = lookup;
    _ = @import("lookup_test.zig");
}
