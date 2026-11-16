use crate::{
    ast::{BinaryOp, Builtin, UnaryOp},
    ir::IR,
};
use std::collections::{HashMap, HashSet};

#[derive(PartialEq, Eq, Hash)]
enum Key {
    Num(u32),
    State(String, usize),
    Tap(String, usize),
    Clock(u32, Option<usize>),
    Phase(usize, usize),
    Pitch(usize),
    Unary(UnaryOp, usize),
    Binary(usize, BinaryOp, usize),
    Builtin(Builtin, Vec<usize>),
    Bar(usize),
    Seq(Vec<usize>, usize),
    Mix(Vec<usize>),
    Unique(usize),
}

#[derive(Default)]
pub(crate) struct Cse {
    ids: HashMap<Key, usize>,
    emitted: HashSet<usize>,
    versions: HashMap<String, usize>,
    phase: Option<usize>,
    voice: usize,
}

impl Cse {
    pub(crate) fn share(&mut self, expr: &IR, voice: usize) -> IR {
        self.voice = voice;
        self.visit(expr).1
    }

    pub(crate) fn store(&mut self, key: &str) {
        *self.versions.entry(key.into()).or_default() += 1;
    }

    fn id(&mut self, key: Key) -> usize {
        let id = self.ids.len();
        *self.ids.entry(key).or_insert(id)
    }

    fn state(&mut self, name: &str, voice: usize) -> (String, usize) {
        let key = format!("{name}/{voice}");
        let version = self.versions.get(&key).copied().unwrap_or(0);
        (key, version)
    }

    fn visit(&mut self, expr: &IR) -> (usize, IR) {
        let (key, node) = match expr {
            IR::Rest => return (self.id(Key::Num(0f32.to_bits())), IR::Rest),
            IR::Num(num) => return (self.id(Key::Num(num.to_bits())), expr.clone()),
            IR::Ref { name, voice: index } => {
                let (key, version) = self.state(name, index.unwrap_or(self.voice));
                return (self.id(Key::State(key, version)), expr.clone());
            }
            IR::Clock(rate) => {
                let key = Key::Clock(rate.to_bits(), self.phase);
                return (self.id(key), expr.clone());
            }
            IR::Tap {
                name,
                voice: index_voice,
                index,
            } => {
                let (id, index) = self.visit(index);
                let key = format!("{name}/{}", index_voice.unwrap_or(self.voice));
                let node = IR::Tap {
                    name: name.clone(),
                    voice: *index_voice,
                    index: Box::new(index),
                };
                (Key::Tap(key, id), node)
            }
            IR::Pitch(pitch) => {
                let (id, pitch) = self.visit(pitch);
                (Key::Pitch(id), IR::Pitch(Box::new(pitch)))
            }
            IR::Unary(op, operand) => {
                let (id, operand) = self.visit(operand);
                (Key::Unary(*op, id), IR::Unary(*op, Box::new(operand)))
            }
            IR::Binary(lhs, op, rhs) => {
                let (lhs_id, lhs) = self.visit(lhs);
                let (rhs_id, rhs) = self.visit(rhs);
                let node = IR::Binary(Box::new(lhs), *op, Box::new(rhs));
                (Key::Binary(lhs_id, *op, rhs_id), node)
            }
            IR::Builtin { kind, args } => {
                // only the arguments that have a default are compiled
                let defaults = kind.defaults();
                let (mut ids, args): (Vec<_>, Vec<_>) = args
                    .iter()
                    .take(defaults.len())
                    .map(|arg| self.visit(arg))
                    .unzip();
                for default in &defaults[ids.len()..] {
                    ids.push(self.id(Key::Num(default.to_bits())));
                }
                let key = match kind {
                    Builtin::Noise | Builtin::Print => Key::Unique(self.ids.len()),
                    Builtin::Bar => Key::Bar(self.state("bpm", 0).1),
                    _ => Key::Builtin(*kind, ids),
                };
                let node = IR::Builtin { kind: *kind, args };
                if defaults.is_empty() {
                    return (self.id(key), node);
                }
                (key, node)
            }
            IR::Seq { items, input } => {
                let (input_id, input) = self.visit(input);
                let phase = self.id(Key::Phase(input_id, items.len()));
                let outer = self.phase.replace(phase);
                let (mut ids, mut items): (Vec<_>, Vec<_>) =
                    items.iter().rev().map(|item| self.visit(item)).unzip();
                self.phase = outer;
                ids.reverse();
                items.reverse();
                let node = IR::Seq {
                    items,
                    input: Box::new(input),
                };
                (Key::Seq(ids, input_id), node)
            }
            IR::Mix(items) => {
                let (ids, items): (Vec<_>, Vec<_>) =
                    items.iter().map(|item| self.visit(item)).unzip();
                (Key::Mix(ids), IR::Mix(items))
            }
            IR::Share(..) | IR::Shared(..) => unreachable!(),
        };
        let id = self.id(key);
        if self.emitted.insert(id) {
            (id, IR::Share(id, Box::new(node)))
        } else {
            (id, IR::Shared(id))
        }
    }
}
