use crate::ast::Expr;

// intervals in semitones above the root
const CHORDS: &[(&str, &[i32])] = &[
    ("", &[0, 4, 7]),
    ("m", &[0, 3, 7]),
    ("dim", &[0, 3, 6]),
    ("aug", &[0, 4, 8]),
    ("sus2", &[0, 2, 7]),
    ("sus4", &[0, 5, 7]),
    ("6", &[0, 4, 7, 9]),
    ("m6", &[0, 3, 7, 9]),
    ("7", &[0, 4, 7, 10]),
    ("maj7", &[0, 4, 7, 11]),
    ("m7", &[0, 3, 7, 10]),
    ("m7b5", &[0, 3, 6, 10]),
    ("dim7", &[0, 3, 6, 9]),
    ("add9", &[0, 4, 7, 14]),
    ("9", &[0, 4, 7, 10, 14]),
    ("maj9", &[0, 4, 7, 11, 14]),
    ("m9", &[0, 3, 7, 10, 14]),
    ("11", &[0, 4, 7, 10, 14, 17]),
    ("m11", &[0, 3, 7, 10, 14, 17]),
    ("13", &[0, 4, 7, 10, 14, 17, 21]),
    ("maj13", &[0, 4, 7, 11, 14, 17, 21]),
    ("m13", &[0, 3, 7, 10, 14, 17, 21]),
];

// midi note number of the root of chords
const CHORD_ROOT: i32 = 48;

// a note with an octave, e.g., `a4`, `c#3`, `eb5`, or a chord with its root in octave 3, e.g.,
// `A`, `F#m7`, `Bbmaj9`, as pitches
pub(crate) fn resolve(name: &str) -> Option<Expr> {
    if name.starts_with(|ch: char| ch.is_ascii_lowercase()) {
        let (class, octave) = root(name)?;
        let octave: i32 = octave.parse().ok()?;
        Some(Expr::Pitch((12 * (octave + 1) + class) as f32))
    } else {
        let (class, quality) = root(name)?;
        let (_, intervals) = CHORDS.iter().find(|(name, _)| *name == quality)?;
        Some(Expr::Stack(
            intervals
                .iter()
                .map(|interval| Expr::Pitch((CHORD_ROOT + class + interval) as f32))
                .collect(),
        ))
    }
}

// splits off the pitch class, in semitones above c, followed by an optional sharp or flat
fn root(name: &str) -> Option<(i32, &str)> {
    let mut chars = name.chars();
    let class = match chars.next()?.to_ascii_lowercase() {
        'c' => 0,
        'd' => 2,
        'e' => 4,
        'f' => 5,
        'g' => 7,
        'a' => 9,
        'b' => 11,
        _ => return None,
    };
    let rest = chars.as_str();
    Some(match rest.chars().next() {
        Some('#') => (class + 1, &rest[1..]),
        Some('b') => (class - 1, &rest[1..]),
        _ => (class, rest),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pitches(name: &str) -> Option<Vec<f32>> {
        match resolve(name)? {
            Expr::Pitch(pitch) => Some(vec![pitch]),
            Expr::Stack(items) => Some(
                items
                    .into_iter()
                    .map(|item| match item {
                        Expr::Pitch(pitch) => pitch,
                        other => panic!("expected a pitch, got {other:?}"),
                    })
                    .collect(),
            ),
            other => panic!("expected pitches, got {other:?}"),
        }
    }

    #[test]
    fn resolve_note() {
        assert_eq!(pitches("a4"), Some(vec![69.0]));
        assert_eq!(pitches("c4"), Some(vec![60.0]));
        assert_eq!(pitches("c#4"), Some(vec![61.0]));
        assert_eq!(pitches("eb4"), Some(vec![63.0]));
        assert_eq!(pitches("b3"), Some(vec![59.0]));
        assert_eq!(pitches("bb3"), Some(vec![58.0]));
        assert_eq!(pitches("c0"), Some(vec![12.0]));

        // notes need an octave
        assert_eq!(pitches("a"), None);
        assert_eq!(pitches("x4"), None);
        assert_eq!(pitches("c4x"), None);
    }

    #[test]
    fn resolve_chord() {
        assert_eq!(pitches("C"), Some(vec![48.0, 52.0, 55.0]));
        assert_eq!(pitches("A"), Some(vec![57.0, 61.0, 64.0]));
        assert_eq!(pitches("Am7"), Some(vec![57.0, 60.0, 64.0, 67.0]));
        assert_eq!(pitches("F#m7"), Some(vec![54.0, 57.0, 61.0, 64.0]));
        assert_eq!(pitches("Bb"), Some(vec![58.0, 62.0, 65.0]));
        assert_eq!(
            pitches("D13"),
            Some(vec![50.0, 54.0, 57.0, 60.0, 64.0, 67.0, 71.0])
        );

        assert_eq!(pitches("Cfoo"), None);
        assert_eq!(pitches("X"), None);
    }
}
