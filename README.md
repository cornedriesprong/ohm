# Ohm

Ohm is a live-coding language/DSL for audio synthesis and processing. It is based on the [Koto](https://koto.dev/) scripting language, which is embedded into a realtime audio synthesis engine, and and can be used to define audio graphs. These graphs can be evaluated and rendered in realtime, allowing for fast iteration and experimentation.

The Koto language has an elegant and minimal syntax. For instance, a simple 2-operator FM synthesizer can we defined in Ohm as:

```
sine(sine(400) * 200 + 400)
```

or, when leaving out the optional parentheses and using Koto's [function piping](https://koto.dev/docs/0.15/language/#function-piping) operator:

```
(sine 400) * 200 + 400 -> sine
```

There is no distinction between audio- and control signals, everything is a signal that can modulate any other signal. Audio graphs constructed in Ohm have single-sample feedback, which allows for creating things like filters and delay-based effects (chorus, flanger, etc.).

Note that re-evaluating the code causes discontinuities in the audio output, since the new graph is reconstructed in place of the old one.

## Usage

The way it works is you start the engine with a path to a koto file as an input parameter:

```
cargo run --release -- examples/sine.koto
```

Ohm will parse the code in the file and, if it's valid, will start playing back the audio. It will listen for any changes in the file and reconstruct the graph on save.

WARNING: Ohm currently doesn't have a built-in limiter, and it's trivial to create a signal that will clip the output. Watch your ears and your speakers!

## Functions

Ohm currently contains the following synthesis functions and effects:

- `sine(freq)` a sine wave.
- `square(freq)` a square wave.
- `saw(freq)` a bandlimited sawtooth wave.
- `pulse(freq)` a pulse wave with a single-sample impulse, primarily useful for triggering `ar`, `env` or `pluck` functions.
- `noise()` white noise, also useful as a randomization source.
- `ar(attack, release, trig)` a simple attack-release envelope, can be triggered by a pulse function.
- `svf(cutoff, resonance, input)` a state-variable lowpass filter.
- `seq(list, trig)` a sequencer that outputs the next value in the list each time it receives a trigger.
- `pipe(delay, input)` delays a signal by *n* samples.
- `pluck(frequency, tone, damping, trig)` a Karplus-Strong pluck string synthesis voice.
- `reverb(frequency, tone, damping, trig)` a FDN reverb.
- `reverb(input)` a feedback-delay-network reverb effect.
- `delay(input)` a delay effect.
- basic arithmetic operators `+`, `-`, `*` and `/` can be applied to any signal.

## Similar projects
- [Faust](https://faust.grame.fr/)
- [Csound](https://csound.com/)
- [SuperCollider](https://supercollider.github.io/)
- [TidalCycles](https://tidalcycles.org/)
- [Sonic Pi](https://sonic-pi.net/)
- [Chuck](http://chuck.stanford.edu/)
- [Glicol](https://glicol.org/)
