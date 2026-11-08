//! Host-only dispatch probes cannot acquire a device or compile a shader.
pub const Device = struct {
    units: bool,
    pub fn tensorUnits(d: Device) bool {
        return d.units;
    }
};
