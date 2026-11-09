use crate::ast::{BinaryOp, Builtin, UnaryOp};

// a single voice of an expanded expression
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IR {
    Rest,
    Num(f32),
    Pitch(Box<IR>),
    Ref {
        name: String,
        voice: Option<usize>,
    },
    Tap {
        name: String,
        voice: Option<usize>,
        index: Box<IR>,
    },
    Clock(f32),
    Unary(UnaryOp, Box<IR>),
    Binary(Box<IR>, BinaryOp, Box<IR>),
    Builtin {
        kind: Builtin,
        args: Vec<IR>,
    },
    Seq {
        items: Vec<IR>,
        input: Box<IR>,
    },
    Mix(Vec<IR>),
}
