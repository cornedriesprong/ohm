use strum_macros::{Display, EnumString};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum UnaryOp {
    Negate,
    Floor,
    Tanh,
    Abs,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Pow,
    Mod,
    Min,
    Max,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Assign {
    pub(crate) names: Vec<String>,
    pub(crate) rhs: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Ast {
    Assign(Assign),
    Expr(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub(crate) enum Builtin {
    Ramp,
    Sin,
    Saw,
    Square,
    Tri,
    Noise,
    Print,
    Lp,
    Bp,
    Hp,
    Mix,
    Clock,
    Floor,
    Tanh,
    Abs,
    Min,
    Max,
    Sr,
    Bar,
}

impl Builtin {
    pub(crate) fn defaults(&self) -> &'static [f32] {
        match self {
            // freq, phase offset
            Builtin::Ramp | Builtin::Sin | Builtin::Saw | Builtin::Square | Builtin::Tri => {
                &[100.0, 0.0]
            }
            // input, cutoff, q
            Builtin::Lp | Builtin::Bp | Builtin::Hp => &[0.0, 1000.0, 0.707],
            Builtin::Noise | Builtin::Mix | Builtin::Clock | Builtin::Sr | Builtin::Bar => &[],
            Builtin::Min | Builtin::Max => &[0.0, 0.0],
            Builtin::Print | Builtin::Floor | Builtin::Tanh | Builtin::Abs => &[0.0],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Rest,
    Num(f32),
    Pitch(f32),
    Ident(String),
    Ref(String),
    Tap {
        name: String,
        index: Box<Expr>,
    },
    Unary(UnaryOp, Box<Expr>),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Fn {
        params: Vec<String>,
        defaults: Vec<Expr>,
        assigns: Vec<Assign>,
        body: Box<Expr>,
    },
    Builtin {
        kind: Builtin,
        args: Vec<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Seq(Vec<Expr>),
    Stack(Vec<Expr>),
}
