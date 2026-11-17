# ohm

> ohm is a live coding language for music and sound.

> [!CAUTION]
> ohm is under active development. Syntax may change, patches written today may not compile later on.

## Run

Requires [Rust](https://rustup.rs).

    git clone git@github.com:cornedriesprong/ohm.git
    cd ohm-rs
    cargo run --release

Edit `patch.ohm` and save.

## Design

ohm was designed with the following goals in mind:

- to allow improvisation
- to have a minimal & expressive syntax
- to be modular and composable

## Language

In ohm every expression yields a *signal*, i.e., a continuous stream of samples. So even just evaluating a single number like `1` will output a constant DC offset value. There is no distinction between audio and control signals. Everything runs at audio rate, and you can send any signal to any input.

### Signals

Depending on their context, signals can perform different functions:

1. Generate sound: `sin 440`
2. Control parameters (either static or modulating): `voice 200 + lfo 0.5 * 20`
3. Logic / control flow
4. Timing / sequencing

### Sequences

Sequencing in ohm is driven by phasors, i.e., signals that ramp from 0 to 1 over a given period. The global `clock` function outputs a phasor derived from the global bpm; for example, set `bpm = 140` to change the tempo.

Multiply or divide a clock signal to derive related rates, e.g., `clock * 2` or `clock / 2`.

To step through values in order, put them in angle brackets: `<200 300 100>`. Sequences are driven by the global `clock` implicitly and can be nested.

### Stacks

Putting expressions in curly braces creates a stack of parallel signals: `saw {200 300}`. The signal graph downstream of the stack will get expanded (duplicated) to the number of channels. This can be used for polyphony as well as stereo or multichannel output. Stacks can be destructured to get their individual elements: `{l r} = saw -> pan 0.2` Stacks can be nested inside sequences or vice versa.

### `->` operator

Compose functions from left to right using the `->` operator: `saw 200 -> lp 1500`. The left side's output will be applied as the first argument to the right side. By convention, functions take an input signal as their first argument.

## Builtin functions

!any parameters at have a default argument (e.g., `freq=100`) are optional and can be omitted.

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

### Sequences

Sequencing throughout ohm happens using unipolar phasor signals, i.e., a signal

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

