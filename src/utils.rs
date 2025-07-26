//! Utility functions

use crate::nodes::Frame;

pub fn freq_to_period(sample_rate: f32, freq: f32) -> f32 {
    sample_rate / freq
}

pub fn cubic_interpolate(buffer: &[Frame], read_head: f32) -> Frame {
    let len = buffer.len() as isize;
    if len == 0 {
        return [0.0; 2];
    }

    let index_floor = read_head.floor() as isize;
    let t = read_head.fract(); // Fractional part for interpolation
    let t2 = t * t;
    let t3 = t2 * t;

    let mut output_frame: Frame = [0.0; 2];

    for ch in 0..2 {
        // Get four neighboring sample indices, using `rem_euclid` for safe wrapping
        let idx0 = (index_floor - 1).rem_euclid(len) as usize;
        let idx1 = index_floor.rem_euclid(len) as usize;
        let idx2 = (index_floor + 1).rem_euclid(len) as usize;
        let idx3 = (index_floor + 2).rem_euclid(len) as usize;

        // Fetch the four points (p0, p1, p2, p3) for the current channel
        let p0 = buffer[idx0][ch];
        let p1 = buffer[idx1][ch];
        let p2 = buffer[idx2][ch];
        let p3 = buffer[idx3][ch];

        // Apply the Catmull-Rom interpolation formula:
        // \[
        // y(t) = 0.5 \times ( (2P_1) + (-P_0+P_2)t + (2P_0-5P_1+4P_2-P_3)t^2 + (-P_0+3P_1-3P_2+P_3)t^3 )
        // \]
        let sample = 0.5 * (
            (2.0 * p1) +
                (-p0 + p2) * t +
                (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        );

        output_frame[ch] = sample;
    }

    output_frame
}
