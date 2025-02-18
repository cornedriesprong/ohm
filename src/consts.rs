//! Constants

// TODO: don't hardcode sample rate
pub const SAMPLE_RATE: f32 = 44100.0;
pub const A4_FREQ: f32 = 440.0;
pub const A4_MIDI: u8 = 69;

/// test signals
pub const DC_SIGNAL: [f32; 7] = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
pub const NYQUIST_SIGNAL: [f32; 7] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
pub const HALF_NYQUIST_SIGNAL: [f32; 7] = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
pub const QUARTER_NYQUIST_SIGNAL: [f32; 9] =
    [0.0, 0.707, 1.0, 0.707, 0.0, -0.707, -1.0, -0.707, 0.0];
pub const IMPULSE_SIGNAL: [f32; 7] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
