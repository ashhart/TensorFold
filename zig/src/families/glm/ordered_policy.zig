//! Arithmetic ownership is a device and rank decision, independent of prompt-cut settings.
pub fn enabled(tensor_units: bool, paired: bool) bool {
    return !tensor_units and !paired;
}
