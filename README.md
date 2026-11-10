# ohm

> ohm is a live coding language for music and sound.

## Run

Requires [Rust](https://rustup.rs).

    git clone git@github.com:cornedriesprong/ohm.git
    cd ohm-rs
    cargo run --release

Edit `patch.ohm` and save.

## Language

Set `bpm = 120` to change the global tempo.

### Oscillators

| name | arguments | output |
|---|---|---|
| `sin` `saw` `square` `tri` | `freq=100 phase=0` | -1..1, band-limited |
| `ramp` | `freq=100 phase=0` | 0..1 |
| `noise` | | white noise, -1..1 |

### Filters

| name | arguments | output |
|---|---|---|
| `lp` `bp` `hp` | `in cutoff=1000 q=0.707` | state variable filter |

### Time

| name | output |
|---|---|
| `clock` | 0..1 ramp per bar. `clock / n` slows it down. inside a rent step |
| `bar` | bar length in seconds |
| `sr` | sample rate |

Set the tempo with `bpm = 180`. Default is 120.

### Math

| name | arguments |
|---|---|
| `floor` `tanh` `abs` | `x` |
| `min` `max` | `a b` |

Operators: `+ - * / ^ %` and `== != > < >= <=`. Comparisons return 0 or 1.

### Voices

| name | arguments | output |
|---|---|---|
| `mix` | any | sums all voices into one |
| `map` | `seq fn` | applies `fn` to each item of a sequence, keeps
| `print` | `x` | prints `x` every sample and passes it through |

### Pitches

Notes: `a4`, `c#3`, `eb5`.
Chords: `Am`, `F#m7`, `Bbmaj9`.
Qualities: `m` `dim` `aug` `sus2` `sus4` `6` `m6` `7` `maj7` `m7` `aj9` `m9` `11` `m11` `13` `maj13` `m13`.

## Prelude

See `prelude.ohm`.

### Generators

| name | arguments | output |
|---|---|---|
| `lfo` | `freq=0.1` | sine, 0..1 |
| `pm` | `freq=100 fb=0.2` | sine with phase feedback |
| `pink` | `amp=1` | pink noise |
| `pluck` | `freq gate decay` | Karplus-Strong plucked string |
| `voice` | `freq cutoff=500` | saw through a lowpass |
| `synth` | `freq` | `voice` with a decay per bar |
| `voss` | `tick c` | 1/f random, roughly -1..1 |

### Effects

| name | arguments | output |
|---|---|---|
| `lp1` | `x coeff=0.001` | one-pole lowpass |
| `avg` | `x coeff=0.5` | two-sample average |
| `comb` | `x n g damp` | damped feedback comb, `n` frames |
| `allpass` | `x n` | allpass filter, `n` frames |
| `rev` | `in size=0.98 damp=0.2` | stereo reverb |
| `bbd` | `in division fb` | Bucket-brigade delay, `division` in bars |
| `pan` | `in pos=0` | pan, -1 (left) to 1 (right) |

### Rhythm and pitch

| name | arguments | output |
|---|---|---|
| `bres` | `k n` | Bresenham algorithm: `k` hits spread over `n` steps per bar |
| `hit` | `k n` | `bres` with a decay envelope |
| `scale` | `x k=7` | maps `x` to a frequency, `k` steps per octave

