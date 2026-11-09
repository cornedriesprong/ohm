use crate::{
    ast::{Assign, Ast, Builtin, Expr},
    compiler::CompileError,
    expand::{expand, storable},
    ir::IR,
    pitch,
};
use std::collections::HashMap;

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Resolved {
    pub(crate) stores: Vec<(String, Vec<IR>)>,
    pub(crate) outputs: Vec<Vec<IR>>,
}

pub(crate) struct Env {
    scopes: Vec<HashMap<String, Expr>>,
    widths: HashMap<String, usize>,
    resolved: Resolved,
    renames: HashMap<String, String>,
    calls: usize,
}

impl Env {
    pub(crate) fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            widths: HashMap::new(),
            resolved: Resolved::default(),
            renames: HashMap::new(),
            calls: 0,
        }
    }

    pub(crate) fn resolve(&mut self, nodes: &[Ast]) -> Result<Resolved, CompileError> {
        for item in nodes {
            self.resolve_item(item)?;
        }
        Ok(std::mem::take(&mut self.resolved))
    }

    fn resolve_item(&mut self, node: &Ast) -> Result<(), CompileError> {
        match node {
            Ast::Assign(Assign { names, rhs }) => {
                let rhs = self.resolve_expr(rhs)?;
                let stores = self.assign(names, names, rhs)?;
                self.resolved.stores.extend(stores);
            }
            Ast::Expr(expr) => {
                let expr = self.resolve_expr(expr)?;
                self.resolved.outputs.push(expand(expr, &self.widths)?);
            }
        }
        Ok(())
    }

    fn assign(
        &mut self,
        names: &[String],
        keys: &[String],
        rhs: Expr,
    ) -> Result<Vec<(String, Vec<IR>)>, CompileError> {
        if let Expr::Fn { .. } = rhs {
            self.scope()
                .extend(names.iter().map(|name| (name.clone(), rhs.clone())));
            return Ok(vec![]);
        }

        for key in keys {
            self.widths.remove(key);
        }
        let voices = expand(rhs.clone(), &self.widths)?;

        let stores: Vec<_> = if let ([name], [key]) = (names, keys) {
            let binding = if refers_to(&rhs, key) || storable(&voices) {
                Expr::Ref(key.clone())
            } else {
                rhs
            };
            self.scope().insert(name.clone(), binding);
            vec![(key.clone(), voices)]
        } else {
            names
                .iter()
                .zip(keys)
                .enumerate()
                .map(|(i, (name, key))| {
                    self.scope().insert(name.clone(), Expr::Ref(key.clone()));
                    let voice = match voices.len() {
                        0 => IR::Rest,
                        len => voices[i % len].clone(),
                    };
                    (key.clone(), vec![voice])
                })
                .collect()
        };

        for (key, voices) in &stores {
            self.widths.insert(key.clone(), voices.len());
        }
        Ok(stores)
    }

    fn resolve_expr(&mut self, expr: &Expr) -> Result<Expr, CompileError> {
        match expr {
            Expr::Ident(name) => self.call_bare(self.resolve_ident(name)?),
            Expr::Unary(op, operand) => Ok(Expr::Unary(*op, self.resolve_expr(operand)?.into())),
            Expr::Binary(lhs, op, rhs) => Ok(Expr::Binary(
                self.resolve_expr(lhs)?.into(),
                *op,
                self.resolve_expr(rhs)?.into(),
            )),
            Expr::Seq(items) => Ok(Expr::Seq(self.resolve_seq(items)?)),
            Expr::Stack(items) => Ok(Expr::Stack(self.resolve_seq(items)?)),
            Expr::Call { callee, args } => match callee.as_ref() {
                Expr::Ident(name) => self.resolve_call(name, args),
                callee => Ok(Expr::Call {
                    callee: self.resolve_expr(callee)?.into(),
                    args: self.resolve_seq(args)?,
                }),
            },
            Expr::Tap { name, index } => Ok(Expr::Tap {
                name: self.resolve_ref(name),
                index: self.resolve_expr(index)?.into(),
            }),
            _ => Ok(expr.clone()),
        }
    }

    fn resolve_ref(&self, name: &str) -> String {
        if let Some(key) = self.renames.get(name) {
            return key.clone();
        }
        if self.scopes.len() > 1
            && let Some(Expr::Ref(key)) = self.scopes.last().unwrap().get(name)
        {
            return key.clone();
        }
        name.into()
    }

    fn resolve_seq(&mut self, items: &[Expr]) -> Result<Vec<Expr>, CompileError> {
        items
            .iter()
            .map(|item| self.resolve_expr(item))
            .collect::<Result<Vec<_>, _>>()
    }

    fn resolve_ident(&self, name: &str) -> Result<Expr, CompileError> {
        if name == "_" {
            Ok(Expr::Rest)
        } else if let Ok(kind) = name.parse::<Builtin>() {
            Ok(Expr::Builtin { kind, args: vec![] })
        } else if let Some(expr) = self
            .scopes
            .last()
            .unwrap()
            .get(name)
            .or(self.scopes[0].get(name))
        {
            Ok(expr.clone())
        } else if let Some(expr) = pitch::resolve(name) {
            Ok(expr)
        } else {
            Err(CompileError::UnresolvedIdentifier(name.into()))
        }
    }

    fn call_bare(&mut self, expr: Expr) -> Result<Expr, CompileError> {
        match expr {
            Expr::Fn {
                ref params,
                ref defaults,
                ..
            } if params.len() == defaults.len() => self.apply(&expr, vec![]),
            expr => Ok(expr),
        }
    }

    fn call_bare_all(&mut self, args: Vec<Expr>) -> Result<Vec<Expr>, CompileError> {
        args.into_iter().map(|arg| self.call_bare(arg)).collect()
    }

    fn resolve_call(&mut self, name: &str, args: &[Expr]) -> Result<Expr, CompileError> {
        let args = args
            .iter()
            .map(|arg| match arg {
                Expr::Ident(name) => self.resolve_ident(name),
                arg => self.resolve_expr(arg),
            })
            .collect::<Result<Vec<_>, _>>()?;

        if let Ok(kind) = name.parse::<Builtin>() {
            let args = self.call_bare_all(args)?;
            return Ok(Expr::Builtin { kind, args });
        };

        if name == "map" {
            let [seq, f]: [Expr; 2] = args
                .try_into()
                .map_err(|args| CompileError::UnexpectedExpression(format!("map {args:?}")))?;
            return self.map(seq, &f);
        }

        match self.resolve_ident(name)? {
            f @ Expr::Fn { .. } => self.apply(&f, args),
            callee => Ok(Expr::Call {
                callee: callee.into(),
                args: self.call_bare_all(args)?,
            }),
        }
    }

    fn apply(&mut self, f: &Expr, args: Vec<Expr>) -> Result<Expr, CompileError> {
        let Expr::Fn {
            params,
            defaults,
            assigns,
            body,
        } = f
        else {
            return Err(CompileError::NotCallable(format!("{f:?}")));
        };
        self.scopes.push(HashMap::new());
        let renames = std::mem::take(&mut self.renames);
        let result = self.bind_params(params, defaults, args, assigns, body);
        self.scopes.pop();
        self.renames = renames;
        result
    }

    fn scope(&mut self) -> &mut HashMap<String, Expr> {
        self.scopes.last_mut().unwrap()
    }

    fn bind_params(
        &mut self,
        params: &[String],
        defaults: &[Expr],
        args: Vec<Expr>,
        assigns: &[Assign],
        body: &Expr,
    ) -> Result<Expr, CompileError> {
        self.calls += 1;
        let prefix = format!("call{}", self.calls);
        let mut args = args.into_iter();
        for (i, param) in params.iter().enumerate() {
            // missing arguments take their default, which sees the preceding parameters
            let arg = match args.next() {
                Some(arg) => arg,
                None => match (i + defaults.len()).checked_sub(params.len()) {
                    Some(j) => self.resolve_expr(&defaults[j])?,
                    None => break,
                },
            };
            let key = format!("{prefix}/{param}");
            let stores =
                self.assign(std::slice::from_ref(param), std::slice::from_ref(&key), arg)?;
            // only stored arguments are referred to
            if self.scope().get(param) == Some(&Expr::Ref(key)) {
                self.resolved.stores.extend(stores);
            }
        }
        self.resolve_body(&prefix, assigns, body)
    }

    // the body's assignments are hoisted and get their own state per call
    fn resolve_body(
        &mut self,
        prefix: &str,
        assigns: &[Assign],
        body: &Expr,
    ) -> Result<Expr, CompileError> {
        // up front, so refs can refer to names assigned later on
        for Assign { names, .. } in assigns {
            self.renames.extend(
                names
                    .iter()
                    .map(|name| (name.clone(), format!("{prefix}/{name}"))),
            );
        }
        for Assign { names, rhs } in assigns {
            let keys: Vec<_> = names
                .iter()
                .map(|name| format!("{prefix}/{name}"))
                .collect();
            let rhs = self.resolve_expr(rhs)?;
            let stores = self.assign(names, &keys, rhs)?;
            self.resolved.stores.extend(stores);
        }
        self.resolve_expr(body)
    }

    // recurses into nested seqs, keeps rests
    fn map(&mut self, expr: Expr, f: &Expr) -> Result<Expr, CompileError> {
        match expr {
            Expr::Seq(items) => Ok(Expr::Seq(
                items
                    .into_iter()
                    .map(|item| self.map(item, f))
                    .collect::<Result<_, _>>()?,
            )),
            // a called seq keeps its input
            Expr::Call { callee, args } if matches!(*callee, Expr::Seq(..)) => Ok(Expr::Call {
                callee: self.map(*callee, f)?.into(),
                args,
            }),
            Expr::Rest => Ok(Expr::Rest),
            item => self.apply(f, vec![item]),
        }
    }
}

fn refers_to(expr: &Expr, key: &str) -> bool {
    match expr {
        Expr::Ref(name) => name == key,
        Expr::Tap { name, index } => name == key || refers_to(index, key),
        Expr::Unary(_, operand) => refers_to(operand, key),
        Expr::Binary(lhs, _, rhs) => refers_to(lhs, key) || refers_to(rhs, key),
        Expr::Builtin { args: items, .. } | Expr::Seq(items) | Expr::Stack(items) => {
            items.iter().any(|item| refers_to(item, key))
        }
        Expr::Call { callee, args } => {
            refers_to(callee, key) || args.iter().any(|arg| refers_to(arg, key))
        }
        Expr::Rest | Expr::Num(..) | Expr::Pitch(..) | Expr::Ident(..) | Expr::Fn { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{BinaryOp, Builtin, UnaryOp},
        lexer::Lexer,
        parser::Parser,
    };

    fn resolve_src(src: &str) -> Result<Resolved, CompileError> {
        let mut parser = Parser::new(Lexer::new(src));
        let items = parser.parse().unwrap();
        let mut env = Env::new();
        env.resolve(&items)
    }

    #[test]
    fn resolves_value_assignment() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("x = 2 + 3")?,
            Resolved {
                stores: vec![(
                    "x".into(),
                    vec![IR::Binary(
                        IR::Num(2.0).into(),
                        BinaryOp::Add,
                        IR::Num(3.0).into()
                    )]
                )],
                outputs: vec![]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_fn_assignment() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("double = |x| x + x\ndouble 3")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Binary(
                    IR::Num(3.0).into(),
                    BinaryOp::Add,
                    IR::Num(3.0).into(),
                )]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_multiple_fn_assignment() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("add1 add2 = |x| x + x\nadd2 3")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Binary(
                    IR::Num(3.0).into(),
                    BinaryOp::Add,
                    IR::Num(3.0).into(),
                )]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_defaults() -> Result<(), CompileError> {
        let mul = |lhs, rhs| IR::Binary(IR::Num(lhs).into(), BinaryOp::Mul, IR::Num(rhs).into());
        // a bare name calls a function whose parameters all have defaults
        assert_eq!(
            resolve_src("lfo = |freq = 2| freq * 3\nlfo")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![mul(2.0, 3.0)]]
            }
        );
        assert_eq!(
            resolve_src("lfo = |freq = 2| freq * 3\nlfo 4")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![mul(4.0, 3.0)]]
            }
        );
        // a default sees the preceding parameters
        assert_eq!(
            resolve_src("f = |x y = x| x * y\nf 5")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![mul(5.0, 5.0)]]
            }
        );
        // passed to a function, it stays a function
        assert_eq!(
            resolve_src("lfo = |freq = 2| freq * 3\ntwice = |g| g 1 + g 4\ntwice lfo")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Binary(
                    mul(1.0, 3.0).into(),
                    BinaryOp::Add,
                    mul(4.0, 3.0).into(),
                )]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_builtin_no_args() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("noise")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Builtin {
                    kind: Builtin::Noise,
                    args: vec![]
                }]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_negation() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("-(square 100)")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Unary(
                    UnaryOp::Negate,
                    IR::Builtin {
                        kind: Builtin::Square,
                        args: vec![IR::Num(100.0)]
                    }
                    .into()
                )]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_cycle() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("x = sin ~x")?,
            Resolved {
                stores: vec![(
                    "x".into(),
                    vec![IR::Builtin {
                        kind: Builtin::Sin,
                        args: vec![IR::Tap {
                            name: "x".into(),
                            voice: None,
                            index: IR::Num(1.0).into(),
                        }]
                    }]
                )],
                outputs: vec![]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_stack_cycle() -> Result<(), CompileError> {
        // self references refer to the voice they're in
        assert_eq!(
            resolve_src("x = {1 2} + ~x")?,
            Resolved {
                stores: vec![(
                    "x".into(),
                    vec![
                        IR::Binary(
                            IR::Num(1.0).into(),
                            BinaryOp::Add,
                            IR::Tap {
                                name: "x".into(),
                                voice: None,
                                index: IR::Num(1.0).into(),
                            }
                            .into()
                        ),
                        IR::Binary(
                            IR::Num(2.0).into(),
                            BinaryOp::Add,
                            IR::Tap {
                                name: "x".into(),
                                voice: None,
                                index: IR::Num(1.0).into(),
                            }
                            .into()
                        )
                    ]
                )],
                outputs: vec![]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_destructuring() -> Result<(), CompileError> {
        // each name stores one voice, plain names refer to the stored voice
        assert_eq!(
            resolve_src("x y = {1 2}\nx")?,
            Resolved {
                stores: vec![
                    ("x".into(), vec![IR::Num(1.0)]),
                    ("y".into(), vec![IR::Num(2.0)])
                ],
                outputs: vec![vec![IR::Ref {
                    name: "x".into(),
                    voice: Some(0)
                }]]
            }
        );

        // fewer voices than names wrap
        assert_eq!(
            resolve_src("x y = 1")?,
            Resolved {
                stores: vec![
                    ("x".into(), vec![IR::Num(1.0)]),
                    ("y".into(), vec![IR::Num(1.0)])
                ],
                outputs: vec![]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_map() -> Result<(), CompileError> {
        // replaces items, recurses into nested seqs and keeps rests
        let double = |n: f32| IR::Binary(IR::Num(n).into(), BinaryOp::Mul, IR::Num(2.0).into());
        let expected = Resolved {
            stores: vec![],
            outputs: vec![vec![IR::Seq {
                items: vec![
                    double(1.0),
                    IR::Seq {
                        items: vec![double(2.0), IR::Rest],
                        input: IR::Clock(1.0).into(),
                    },
                ],
                input: IR::Clock(1.0).into(),
            }]],
        };
        assert_eq!(resolve_src("map <1 <2 _>> |n| n * 2")?, expected);
        assert_eq!(resolve_src("<1 <2 _>> -> map |n| n * 2")?, expected);

        // items can be stacks
        assert_eq!(
            resolve_src("map <{1 2}> |n| clock")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Seq {
                    items: vec![IR::Clock(1.0)],
                    input: IR::Clock(1.0).into(),
                }]]
            }
        );

        assert!(matches!(
            resolve_src("map <1> 2"),
            Err(CompileError::NotCallable(_))
        ));

        Ok(())
    }

    #[test]
    fn resolve_block() -> Result<(), CompileError> {
        // body state is hoisted and unique per call, the cycle's name refers to its stored value
        let src = "f = |a|\n  x = ~x + a\n  x\n{(f 1) (f 2)}";
        let store = |name: &str, a| {
            (
                name.to_string(),
                vec![IR::Binary(
                    IR::Tap {
                        name: name.into(),
                        voice: None,
                        index: IR::Num(1.0).into(),
                    }
                    .into(),
                    BinaryOp::Add,
                    IR::Num(a).into(),
                )],
            )
        };
        let load = |name: &str| IR::Ref {
            name: name.into(),
            voice: Some(0),
        };
        assert_eq!(
            resolve_src(src)?,
            Resolved {
                stores: vec![store("call1/x", 1.0), store("call2/x", 2.0)],
                outputs: vec![vec![load("call1/x"), load("call2/x")]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_cycle_binding() -> Result<(), CompileError> {
        // a cycle's name refers to its stored value
        assert_eq!(
            resolve_src("x = ~x + 1\nx")?,
            Resolved {
                stores: vec![(
                    "x".into(),
                    vec![IR::Binary(
                        IR::Tap {
                            name: "x".into(),
                            voice: None,
                            index: IR::Num(1.0).into(),
                        }
                        .into(),
                        BinaryOp::Add,
                        IR::Num(1.0).into()
                    )]
                )],
                outputs: vec![vec![IR::Ref {
                    name: "x".into(),
                    voice: Some(0)
                }]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_fn_globals() -> Result<(), CompileError> {
        let double = |n: f32| IR::Binary(IR::Num(n).into(), BinaryOp::Mul, IR::Num(2.0).into());

        // function bodies see global bindings
        assert_eq!(
            resolve_src("double = |x| x * 2\nf = |y| double y\nf 4")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![double(4.0)]]
            }
        );
        assert_eq!(
            resolve_src("double = |x| x * 2\nmap <1> |n| double n")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Seq {
                    items: vec![double(1.0)],
                    input: IR::Clock(1.0).into()
                }]]
            }
        );

        // parameters shadow globals
        assert_eq!(
            resolve_src("x = 1\nf = |x| x\nf 2")?,
            Resolved {
                stores: vec![("x".into(), vec![IR::Num(1.0)])],
                outputs: vec![vec![IR::Num(2.0)]]
            }
        );

        // but not the caller's scope
        assert!(matches!(
            resolve_src("f = |a| b\ng = |b| f 1\ng 2"),
            Err(CompileError::UnresolvedIdentifier(name)) if name == "b"
        ));

        Ok(())
    }

    #[test]
    fn resolve_pitch() -> Result<(), CompileError> {
        let pitch = |n: f32| IR::Pitch(IR::Num(n).into());
        assert_eq!(
            resolve_src("a4")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![pitch(69.0)]]
            }
        );
        // chords are stacks
        assert_eq!(
            resolve_src("Am")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![pitch(57.0), pitch(60.0), pitch(64.0)]]
            }
        );
        // bindings shadow pitches
        assert_eq!(
            resolve_src("a4 = 1\na4")?,
            Resolved {
                stores: vec![("a4".into(), vec![IR::Num(1.0)])],
                outputs: vec![vec![IR::Num(1.0)]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_stack_ref() -> Result<(), CompileError> {
        // refs expand into a voice per stored voice
        assert_eq!(
            resolve_src("x = {1 2}\n~x")?,
            Resolved {
                stores: vec![("x".into(), vec![IR::Num(1.0), IR::Num(2.0)])],
                outputs: vec![vec![
                    IR::Tap {
                        name: "x".into(),
                        voice: Some(0),
                        index: IR::Num(1.0).into(),
                    },
                    IR::Tap {
                        name: "x".into(),
                        voice: Some(1),
                        index: IR::Num(1.0).into(),
                    }
                ]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_scope() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("x = 5\nadd = |y| ~x + y\nadd 5")?,
            Resolved {
                stores: vec![("x".into(), vec![IR::Num(5.0)])],
                outputs: vec![vec![IR::Binary(
                    IR::Tap {
                        name: "x".into(),
                        voice: Some(0),
                        index: IR::Num(1.0).into(),
                    }
                    .into(),
                    BinaryOp::Add,
                    IR::Num(5.0).into(),
                )]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_shadowing() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("x = 5\nadd = |x| x + 1\nadd 3")?,
            Resolved {
                stores: vec![("x".into(), vec![IR::Num(5.0)])],
                outputs: vec![vec![IR::Binary(
                    IR::Num(3.0).into(),
                    BinaryOp::Add,
                    IR::Num(1.0).into(),
                )]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_seq() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("<100 200 _>")?,
            // uncalled seq is driven by the ambient phase
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Seq {
                    items: vec![IR::Num(100.0), IR::Num(200.0), IR::Rest],
                    input: IR::Clock(1.0).into()
                }]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_seq_call() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("x = <100 200>\n0 -> x")?,
            Resolved {
                stores: vec![(
                    "x".into(),
                    vec![IR::Seq {
                        items: vec![IR::Num(100.0), IR::Num(200.0)],
                        input: IR::Clock(1.0).into()
                    }]
                )],
                outputs: vec![
                    // calls to seqs fall through to the VM
                    vec![IR::Seq {
                        items: vec![IR::Num(100.0), IR::Num(200.0)],
                        input: IR::Num(0.0).into()
                    }]
                ]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_stack() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("{100 200}")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Num(100.0), IR::Num(200.0)]]
            }
        );

        Ok(())
    }

    #[test]
    fn dont_shadow_builtins() -> Result<(), CompileError> {
        assert_eq!(
            resolve_src("sin = |x| x\nsin 100")?,
            Resolved {
                stores: vec![],
                outputs: vec![vec![IR::Builtin {
                    kind: Builtin::Sin,
                    args: vec![IR::Num(100.0)]
                }]]
            }
        );

        Ok(())
    }

    #[test]
    fn resolve_invalid_identifier() {
        match resolve_src("invalid 2") {
            Err(CompileError::UnresolvedIdentifier(name)) => assert_eq!(name, "invalid"),
            other => panic!("Expected unresolved identifier, got {other:?}"),
        }
    }

    #[test]
    fn reject_stack_call() {
        // can't call a number
        assert!(matches!(
            resolve_src("0 -> 1"),
            Err(CompileError::NotCallable(_))
        ));

        // can't call a function definition
        assert!(matches!(
            resolve_src("0 -> |x| x + x"),
            Err(CompileError::NotCallable(_))
        ));

        // can't call a stack
        assert!(matches!(
            resolve_src("0 -> {1 2}"),
            Err(CompileError::NotCallable(_))
        ));
        assert!(matches!(
            resolve_src("x = {1 2}\n0 -> x"),
            Err(CompileError::NotCallable(_))
        ));
    }
}
