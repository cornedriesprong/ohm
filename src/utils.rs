//! Utility functions
use crate::consts::{A4_FREQ, A4_MIDI};

pub fn pitch_to_freq(pitch: u8) -> f32 {
    A4_FREQ * (2f32).powf((pitch as f32 - A4_MIDI as f32) as f32 / 12.0)
}

pub fn freq_to_pitch(freq: f32) -> u8 {
    ((freq / A4_FREQ).log2() * 12.0 + A4_MIDI as f32).round() as u8
}

pub fn freq_to_period(sample_rate: f32, freq: f32) -> f32 {
    sample_rate / freq
}

pub fn scale_log(value: f32, min: f32, max: f32) -> f32 {
    min * (max / min).powf(value)
}

pub fn lerp(x: f32, length: f32) -> f32 {
    x / length
}

pub fn xerp(x: f32, length: f32, pow: i8) -> f32 {
    (x / length).powf(pow as f32)
}

pub fn lin_to_log(lin: f32, lin_min: f32, lin_max: f32, min_log: f32, max_log: f32) -> f32 {
    let lin_norm = (lin - lin_min) / (lin_max - lin_min);
    return min_log * (max_log / min_log).powf(lin_norm);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midi_to_freq() {
        assert_eq!(pitch_to_freq(0), 8.175798);
        assert_eq!(pitch_to_freq(69), 440.0);
        assert_eq!(pitch_to_freq(127), 12543.855);
    }

    #[test]
    fn test_freq_to_midi() {
        assert_eq!(freq_to_pitch(8.17), 0);
        assert_eq!(freq_to_pitch(440.0), 69);
        assert_eq!(freq_to_pitch(12543.855), 127);
    }
}
