use crate::{
    ast::{Assign, Ast, BinaryOp, Expr, UnaryOp},
    lexer::{LexError, Lexer, Token},
};

#[derive(Debug)]
pub(crate) enum ParseError {
    LexError(LexError),
    ExpectedIdent(Expr),
    ExpectedPrimary(Token),
    InvalidAssignment(Expr),
    NonTopLevelAssignment,
    InvalidBlock,
    UnclosedSeq(Token),
    UnclosedIndex(Token),
    MissingDefault(String),
}

impl From<LexError> for ParseError {
    fn from(error: LexError) -> Self {
        ParseError::LexError(error)
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::LexError(e) => write!(f, "lex error: {e:?}"),
            ParseError::ExpectedIdent(expr) => write!(f, "expected identifier, got {expr:?}"),
            ParseError::ExpectedPrimary(t) => write!(f, "expected primary expression, got {t:?}"),
            ParseError::InvalidAssignment(expr) => write!(f, "can't assign to {expr:?}"),
            ParseError::NonTopLevelAssignment => {
                write!(f, "assignment is only allowed at the start of a line")
            }
            ParseError::UnclosedSeq(t) => write!(
                f,
                "expected `>` to close a seq, got {t:?}. compare names in parentheses, e.g., `(x) < 1`"
            ),
            ParseError::UnclosedIndex(t) => write!(f, "expected `]` to close an index, got {t:?}"),
            ParseError::MissingDefault(name) => {
                write!(f, "parameter `{name}` follows a default, so needs one")
            }
            ParseError::InvalidBlock => {
                write!(
                    f,
                    "a function body consists of assignments followed by a result"
                )
            }
        }
    }
}

impl std::error::Error for ParseError {}

const PIPE_BP: u8 = 1;

// returns the operator with its left and right binding power
fn binary_op(token: &Token) -> Option<(BinaryOp, u8, u8)> {
    Some(match token {
        Token::Equal => (BinaryOp::Equal, 2, 3),
        Token::NotEqual => (BinaryOp::NotEqual, 2, 3),
        Token::GreaterThan => (BinaryOp::GreaterThan, 2, 3),
        Token::LessThan => (BinaryOp::LessThan, 2, 3),
        Token::GreaterThanOrEqual => (BinaryOp::GreaterThanOrEqual, 2, 3),
        Token::LessThanOrEqual => (BinaryOp::LessThanOrEqual, 2, 3),
        Token::Plus => (BinaryOp::Add, 4, 5),
        Token::Minus => (BinaryOp::Sub, 4, 5),
        Token::Asterisk => (BinaryOp::Mul, 6, 7),
        Token::Slash => (BinaryOp::Div, 6, 7),
        Token::Percent => (BinaryOp::Mod, 6, 7),
        // right associative
        Token::Caret => (BinaryOp::Pow, 9, 8),
        _ => return None,
    })
}

pub(crate) struct Parser<'a> {
    lexer: Lexer<'a>,
    cur: Token,
    peeked: Option<Token>,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(lexer: Lexer<'a>) -> Self {
        Self {
            lexer,
            cur: Token::Eof,
            peeked: None,
        }
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.cur = match self.peeked.take() {
            Some(token) => token,
            None => self.lexer.next()?,
        };
        Ok(())
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let expr = match self.cur {
            Token::Num(num) => {
                self.advance()?;
                Expr::Num(num)
            }
            Token::Ident(ref name) => {
                let expr = Expr::Ident(name.clone());
                self.advance()?;
                expr
            }
            Token::Ref(ref name) => {
                let name = name.clone();
                self.advance()?;
                match self.parse_index()? {
                    Some(index) => Expr::Tap {
                        name,
                        index: index.into(),
                    },
                    None => Expr::Tap {
                        name,
                        index: Expr::Num(1.0).into(),
                    },
                }
            }
            Token::Minus => {
                self.advance()?;
                let expr = self.parse_primary()?;
                Expr::Unary(UnaryOp::Negate, expr.into())
            }
            Token::Bar => {
                self.advance()?;
                self.parse_fn()?
            }
            Token::LParen => {
                self.advance()?; // eat (
                let expr = self.parse_expr()?;
                self.advance()?; // eat )
                expr
            }
            Token::LessThan => {
                self.advance()?; // eat <
                let expr = Expr::Seq(self.parse_primaries(true)?);
                if self.cur != Token::GreaterThan {
                    return Err(ParseError::UnclosedSeq(self.cur.clone()));
                }
                self.advance()?; // eat >
                expr
            }
            Token::LCurly => {
                self.advance()?; // eat {
                let expr = Expr::Stack(self.parse_primaries(true)?);
                self.advance()?; // eat }
                expr
            }
            _ => return Err(ParseError::ExpectedPrimary(self.cur.clone())),
        };

        Ok(expr)
    }

    // `[index]` following a name
    fn parse_index(&mut self) -> Result<Option<Expr>, ParseError> {
        if self.cur != Token::LBracket {
            return Ok(None);
        }
        self.advance()?; // eat [
        let index = self.parse_expr()?;
        if self.cur != Token::RBracket {
            return Err(ParseError::UnclosedIndex(self.cur.clone()));
        }
        self.advance()?; // eat ]
        Ok(Some(index))
    }

    // arguments can't be negated, so `x - 1` is a subtraction rather than a call
    fn parse_primaries(&mut self, negation: bool) -> Result<Vec<Expr>, ParseError> {
        let mut items = Vec::new();
        while matches!(
            self.cur,
            Token::Ident(..)
                | Token::Ref(..)
                | Token::Num(..)
                | Token::LParen
                | Token::LessThan
                | Token::LCurly
                | Token::Bar
        ) || (negation && self.cur == Token::Minus)
        {
            items.push(self.parse_primary()?);
        }
        Ok(items)
    }

    fn parse_names(&self, expr: Expr) -> Result<Vec<String>, ParseError> {
        match expr {
            Expr::Ident(name) => Ok(vec![name]),
            // if we find multiple leading identifiers (e.g., `l r = 1`)
            // they will be parsed as an Expr::Call, and we can extract the names
            Expr::Call { callee, args } => {
                let Expr::Ident(name) = *callee else {
                    return Err(ParseError::InvalidAssignment(*callee));
                };
                let mut names = vec![name];
                for arg in args {
                    let Expr::Ident(name) = arg else {
                        return Err(ParseError::InvalidAssignment(arg));
                    };
                    names.push(name);
                }
                Ok(names)
            }
            // destructuring, e.g., `{l r} = 1`
            Expr::Stack(items) => items
                .into_iter()
                .map(|item| match item {
                    Expr::Ident(name) => Ok(name),
                    other => Err(ParseError::InvalidAssignment(other)),
                })
                .collect(),
            other => Err(ParseError::ExpectedIdent(other)),
        }
    }

    fn parse_call(&mut self) -> Result<Expr, ParseError> {
        // a parenthesized name, e.g. `(x)`, isn't called
        let callable = matches!(self.cur, Token::Ident(..));
        match self.parse_primary()? {
            Expr::Ident(name) if callable => {
                let args = self.parse_primaries(false)?;
                if args.is_empty() {
                    Ok(Expr::Ident(name))
                } else {
                    Ok(Expr::Call {
                        callee: Expr::Ident(name).into(),
                        args,
                    })
                }
            }
            lhs => Ok(lhs),
        }
    }

    fn parse_fn(&mut self) -> Result<Expr, ParseError> {
        let mut params = Vec::new();
        let mut defaults = Vec::new();
        while self.cur != Token::Bar {
            match self.parse_primary()? {
                Expr::Ident(name) => params.push(name),
                other => return Err(ParseError::ExpectedIdent(other)),
            }
            // `name = default`, a primary, so it doesn't take the next parameter as an argument
            if self.cur == Token::Assignment {
                self.advance()?; // eat =
                defaults.push(self.parse_primary()?);
            } else if !defaults.is_empty() {
                return Err(ParseError::MissingDefault(params.pop().unwrap()));
            }
        }
        self.advance()?; // eat closing `|`
        let (assigns, body) = if self.cur == Token::Newline {
            self.parse_block()?
        } else {
            (vec![], self.parse_expr()?)
        };
        Ok(Expr::Fn {
            params,
            defaults,
            assigns,
            body: body.into(),
        })
    }

    // indented lines: assignments followed by a result
    fn parse_block(&mut self) -> Result<(Vec<Assign>, Expr), ParseError> {
        let mut assigns = Vec::new();
        let mut result = None;
        while self.cur == Token::Newline {
            self.advance()?;
            if self.cur == Token::Indent {
                self.advance()?;
                if result.is_some() {
                    return Err(ParseError::InvalidBlock);
                }
                match self.parse_item()? {
                    Ast::Assign(assign) => assigns.push(assign),
                    Ast::Expr(expr) => result = Some(expr),
                }
            } else if self.cur != Token::Newline {
                // the line after the block starts a new item
                self.peeked = Some(std::mem::replace(&mut self.cur, Token::Newline));
                break;
            }
        }
        Ok((assigns, result.ok_or(ParseError::InvalidBlock)?))
    }

    fn parse_pipe(&mut self, lhs: Expr) -> Result<Expr, ParseError> {
        self.advance()?; // eat ->

        let callee = self.parse_primary()?;
        let args = match callee {
            Expr::Ident(..) => {
                let mut args = vec![lhs];
                args.extend(self.parse_primaries(false)?);
                args
            }
            _ => vec![lhs],
        };
        Ok(Expr::Call {
            callee: callee.into(),
            args,
        })
    }

    // precedence climbing: only binds operators with a left binding power of at least `min_bp`.
    // function application binds tighter than any operator, `->` binds loosest
    fn parse_expr_tail(&mut self, mut lhs: Expr, min_bp: u8) -> Result<Expr, ParseError> {
        loop {
            match self.cur {
                Token::Assignment => return Err(ParseError::NonTopLevelAssignment),
                Token::Arrow => {
                    if PIPE_BP < min_bp {
                        break;
                    }
                    lhs = self.parse_pipe(lhs)?;
                }
                _ => {
                    let Some((op, l_bp, r_bp)) = binary_op(&self.cur) else {
                        break;
                    };
                    if l_bp < min_bp {
                        break;
                    }
                    self.advance()?;
                    let rhs = self.parse_call()?;
                    let rhs = self.parse_expr_tail(rhs, r_bp)?;
                    lhs = Expr::Binary(lhs.into(), op, rhs.into());
                }
            }
        }
        Ok(lhs)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_call()?;
        self.parse_expr_tail(lhs, 0)
    }

    fn parse_item(&mut self) -> Result<Ast, ParseError> {
        let lhs = self.parse_call()?;
        if matches!(self.cur, Token::Assignment) {
            self.advance()?;
            let names = self.parse_names(lhs)?;
            let rhs = self.parse_expr()?;
            Ok(Ast::Assign(Assign { names, rhs }))
        } else {
            Ok(Ast::Expr(self.parse_expr_tail(lhs, 0)?))
        }
    }

    pub fn parse(&mut self) -> Result<Vec<Ast>, ParseError> {
        let mut items = Vec::new();
        self.advance()?; // prime parser

        while self.cur != Token::Eof {
            if matches!(self.cur, Token::Newline | Token::Indent) {
                self.advance()?;
                continue;
            }
            items.push(self.parse_item()?);
        }

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_src(src: &str) -> Result<Vec<Ast>, ParseError> {
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        parser.parse()
    }

    #[test]
    fn parse_num() -> Result<(), ParseError> {
        assert_eq!(parse_src("1")?, vec![Ast::Expr(Expr::Num(1.0))]);
        assert_eq!(parse_src("1.5")?, vec![Ast::Expr(Expr::Num(1.5))]);
        assert_eq!(
            parse_src("-1")?,
            vec![Ast::Expr(Expr::Unary(
                UnaryOp::Negate,
                Expr::Num(1.0).into()
            ))]
        );
        assert_eq!(
            parse_src("-1.5")?,
            vec![Ast::Expr(Expr::Unary(
                UnaryOp::Negate,
                Expr::Num(1.5).into()
            ))]
        );

        Ok(())
    }

    #[test]
    fn parse_negation() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("x = 5\n-x")?,
            vec![
                Ast::Assign(Assign {
                    names: vec!["x".into()],
                    rhs: Expr::Num(5.0)
                }),
                Ast::Expr(Expr::Unary(UnaryOp::Negate, Expr::Ident("x".into()).into()))
            ]
        );

        Ok(())
    }

    #[test]
    fn parse_assignment() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("a = 2 + 3")?,
            vec![Ast::Assign(Assign {
                names: vec!["a".into()],
                rhs: Expr::Binary(Expr::Num(2.0).into(), BinaryOp::Add, Expr::Num(3.0).into()),
            })]
        );
        Ok(())
    }

    #[test]
    fn reject_nested_assignment() {
        assert!(matches!(
            parse_src("a = (b = 2)"),
            Err(ParseError::NonTopLevelAssignment)
        ));
        assert!(matches!(
            parse_src("1 + (b = 2)"),
            Err(ParseError::NonTopLevelAssignment)
        ));
    }

    #[test]
    fn parse_ident() -> Result<(), ParseError> {
        assert_eq!(parse_src("x")?, vec![Ast::Expr(Expr::Ident("x".into()))]);

        Ok(())
    }

    #[test]
    fn parse_binop() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("1 + 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Add,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("2 - 1")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(2.0).into(),
                BinaryOp::Sub,
                Expr::Num(1.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 * 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Mul,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 / 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Div,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 == 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Equal,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 != 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::NotEqual,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 > 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::GreaterThan,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 < 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::LessThan,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 >= 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::GreaterThanOrEqual,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 <= 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::LessThanOrEqual,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 ^ 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Pow,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 % 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Mod,
                Expr::Num(2.0).into()
            ))]
        );

        Ok(())
    }

    #[test]
    fn parse_missing_spaces() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("1+2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Add,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1 +2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Add,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1+ 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Add,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1-2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Sub,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("1!=2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::NotEqual,
                Expr::Num(2.0).into()
            ))]
        );

        Ok(())
    }

    #[test]
    fn parse_precedence() -> Result<(), ParseError> {
        // function application binds tighter than operators
        assert_eq!(
            parse_src("sin 100 * 0.5")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Call {
                    callee: Expr::Ident("sin".into()).into(),
                    args: vec![Expr::Num(100.0)]
                }
                .into(),
                BinaryOp::Mul,
                Expr::Num(0.5).into()
            ))]
        );

        assert_eq!(
            parse_src("1 + 2 * 3 ^ 10")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Add,
                Expr::Binary(
                    Expr::Num(2.0).into(),
                    BinaryOp::Mul,
                    Expr::Binary(Expr::Num(3.0).into(), BinaryOp::Pow, Expr::Num(10.0).into())
                        .into()
                )
                .into()
            ))]
        );

        assert_eq!(
            parse_src("2 * 3 + 4")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Binary(Expr::Num(2.0).into(), BinaryOp::Mul, Expr::Num(3.0).into()).into(),
                BinaryOp::Add,
                Expr::Num(4.0).into()
            ))]
        );

        // left associative
        assert_eq!(
            parse_src("1 - 2 - 3")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Binary(Expr::Num(1.0).into(), BinaryOp::Sub, Expr::Num(2.0).into()).into(),
                BinaryOp::Sub,
                Expr::Num(3.0).into()
            ))]
        );

        // right associative
        assert_eq!(
            parse_src("2 ^ 3 ^ 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(2.0).into(),
                BinaryOp::Pow,
                Expr::Binary(Expr::Num(3.0).into(), BinaryOp::Pow, Expr::Num(2.0).into()).into()
            ))]
        );

        // comparisons bind looser than arithmetic
        assert_eq!(
            parse_src("1 + 1 == 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Binary(Expr::Num(1.0).into(), BinaryOp::Add, Expr::Num(1.0).into()).into(),
                BinaryOp::Equal,
                Expr::Num(2.0).into()
            ))]
        );

        // pipe binds loosest, operators after it apply to its result
        assert_eq!(
            parse_src("1 + 2 -> sin * 3")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Call {
                    callee: Expr::Ident("sin".into()).into(),
                    args: vec![Expr::Binary(
                        Expr::Num(1.0).into(),
                        BinaryOp::Add,
                        Expr::Num(2.0).into()
                    )]
                }
                .into(),
                BinaryOp::Mul,
                Expr::Num(3.0).into()
            ))]
        );

        Ok(())
    }

    #[test]
    fn parse_fn() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("fn = || a")?,
            vec![Ast::Assign(Assign {
                names: vec!["fn".into()],
                rhs: Expr::Fn {
                    params: vec![],
                    defaults: vec![],
                    assigns: vec![],
                    body: Expr::Ident("a".into()).into(),
                }
            })]
        );
        assert_eq!(
            parse_src("square = |a| a * a")?,
            vec![Ast::Assign(Assign {
                names: vec!["square".into()],
                rhs: Expr::Fn {
                    params: vec!["a".into()],
                    defaults: vec![],
                    assigns: vec![],
                    body: Expr::Binary(
                        Expr::Ident("a".into()).into(),
                        BinaryOp::Mul,
                        Expr::Ident("a".into()).into()
                    )
                    .into(),
                }
            })]
        );

        Ok(())
    }

    #[test]
    fn parse_fn_defaults() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("f = |a b = 1 c = x| a")?,
            vec![Ast::Assign(Assign {
                names: vec!["f".into()],
                rhs: Expr::Fn {
                    params: vec!["a".into(), "b".into(), "c".into()],
                    defaults: vec![Expr::Num(1.0), Expr::Ident("x".into())],
                    assigns: vec![],
                    body: Expr::Ident("a".into()).into(),
                }
            })]
        );

        // a parameter without a default can't follow one with a default
        assert!(matches!(
            parse_src("f = |a = 1 b| a"),
            Err(ParseError::MissingDefault(name)) if name == "b"
        ));

        Ok(())
    }

    #[test]
    fn parse_call() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("sin 100")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("sin".into()).into(),
                args: vec![Expr::Num(100.0)]
            })]
        );
        assert_eq!(
            parse_src("sin 100 0.5")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("sin".into()).into(),
                args: vec![Expr::Num(100.0), Expr::Num(0.5)]
            })]
        );
        assert_eq!(
            parse_src("x = 100\nsin x 0.5")?,
            vec![
                Ast::Assign(Assign {
                    names: vec!["x".into()],
                    rhs: Expr::Num(100.0)
                }),
                Ast::Expr(Expr::Call {
                    callee: Expr::Ident("sin".into()).into(),
                    args: vec![Expr::Ident("x".into()), Expr::Num(0.5)]
                })
            ]
        );
        Ok(())
    }

    #[test]
    fn parse_block() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("f = |x|\n  y = x\n\n  y\nf 1")?,
            vec![
                Ast::Assign(Assign {
                    names: vec!["f".into()],
                    rhs: Expr::Fn {
                        params: vec!["x".into()],
                        defaults: vec![],
                        assigns: vec![Assign {
                            names: vec!["y".into()],
                            rhs: Expr::Ident("x".into())
                        }],
                        body: Expr::Ident("y".into()).into()
                    }
                }),
                Ast::Expr(Expr::Call {
                    callee: Expr::Ident("f".into()).into(),
                    args: vec![Expr::Num(1.0)]
                })
            ]
        );

        // must end with a result
        assert!(matches!(
            parse_src("f = |x|\n  y = x"),
            Err(ParseError::InvalidBlock)
        ));
        // only the result can be an expression
        assert!(matches!(
            parse_src("f = |x|\n  x\n  x"),
            Err(ParseError::InvalidBlock)
        ));

        Ok(())
    }

    #[test]
    fn parse_seq_brackets() -> Result<(), ParseError> {
        // nested seqs
        assert_eq!(
            parse_src("<1 <2 3>>")?,
            vec![Ast::Expr(Expr::Seq(vec![
                Expr::Num(1.0),
                Expr::Seq(vec![Expr::Num(2.0), Expr::Num(3.0)])
            ]))]
        );
        // stacks in seqs
        assert_eq!(
            parse_src("<{1 2} 3>")?,
            vec![Ast::Expr(Expr::Seq(vec![
                Expr::Stack(vec![Expr::Num(1.0), Expr::Num(2.0)]),
                Expr::Num(3.0)
            ]))]
        );
        // seq arguments
        assert_eq!(
            parse_src("saw <1 2>")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("saw".into()).into(),
                args: vec![Expr::Seq(vec![Expr::Num(1.0), Expr::Num(2.0)])]
            })]
        );
        // a seq followed by a pipe, with or without space
        for src in ["<1 2> -> sin", "<1 2>->sin"] {
            assert_eq!(
                parse_src(src)?,
                vec![Ast::Expr(Expr::Call {
                    callee: Expr::Ident("sin".into()).into(),
                    args: vec![Expr::Seq(vec![Expr::Num(1.0), Expr::Num(2.0)])]
                })]
            );
        }
        // comparisons after a number or parentheses
        assert_eq!(
            parse_src("1 < 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::LessThan,
                Expr::Num(2.0).into()
            ))]
        );
        assert_eq!(
            parse_src("(x) < 2")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Ident("x".into()).into(),
                BinaryOp::LessThan,
                Expr::Num(2.0).into()
            ))]
        );
        // after a name, `<` opens a seq
        assert!(matches!(
            parse_src("x < 2"),
            Err(ParseError::UnclosedSeq(Token::Eof))
        ));

        Ok(())
    }

    #[test]
    fn parse_subtraction() -> Result<(), ParseError> {
        // a minus after a name is a subtraction, not a negated argument
        assert_eq!(
            parse_src("x - 1")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Ident("x".into()).into(),
                BinaryOp::Sub,
                Expr::Num(1.0).into()
            ))]
        );
        // negative arguments need parentheses
        assert_eq!(
            parse_src("sin (-1)")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("sin".into()).into(),
                args: vec![Expr::Unary(UnaryOp::Negate, Expr::Num(1.0).into())]
            })]
        );
        // seq items can still be negated
        assert_eq!(
            parse_src("<1 -1>")?,
            vec![Ast::Expr(Expr::Seq(vec![
                Expr::Num(1.0),
                Expr::Unary(UnaryOp::Negate, Expr::Num(1.0).into())
            ]))]
        );

        Ok(())
    }

    #[test]
    fn parse_fn_arg() -> Result<(), ParseError> {
        // a function argument takes the rest of the expression as its body
        assert_eq!(
            parse_src("map x |n| n * 2")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("map".into()).into(),
                args: vec![
                    Expr::Ident("x".into()),
                    Expr::Fn {
                        params: vec!["n".into()],
                        defaults: vec![],
                        assigns: vec![],
                        body: Expr::Binary(
                            Expr::Ident("n".into()).into(),
                            BinaryOp::Mul,
                            Expr::Num(2.0).into()
                        )
                        .into()
                    }
                ]
            })]
        );

        Ok(())
    }

    #[test]
    fn parse_pipe() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("100 -> sin")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("sin".into()).into(),
                args: vec![Expr::Num(100.0)]
            })]
        );
        assert_eq!(
            parse_src("100 -> sin 0.5")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("sin".into()).into(),
                args: vec![Expr::Num(100.0), Expr::Num(0.5)]
            })]
        );
        assert_eq!(
            parse_src("0 -> <100 200>")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Seq(vec![Expr::Num(100.0), Expr::Num(200.0)]).into(),
                args: vec![Expr::Num(0.0)]
            })]
        );

        Ok(())
    }

    #[test]
    fn parse_chained_pipe() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("100 -> sin -> print")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("print".into()).into(),
                args: vec![Expr::Call {
                    callee: Expr::Ident("sin".into()).into(),
                    args: vec![Expr::Num(100.0)]
                }]
            })]
        );

        Ok(())
    }

    #[test]
    fn parse_parens() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("(1 * 2)")?,
            vec![Ast::Expr(Expr::Binary(
                Expr::Num(1.0).into(),
                BinaryOp::Mul,
                Expr::Num(2.0).into()
            ))]
        );

        Ok(())
    }

    #[test]
    fn parse_destructuring() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("l r = 1")?,
            vec![Ast::Assign(Assign {
                names: vec!["l".to_string(), "r".to_string()],
                rhs: Expr::Num(1.0)
            })]
        );

        Ok(())
    }

    #[test]
    fn parse_stack_destructuring() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("{x y} = saw {100 200}")?,
            vec![Ast::Assign(Assign {
                names: vec!["x".into(), "y".into()],
                rhs: Expr::Call {
                    callee: Expr::Ident("saw".into()).into(),
                    args: vec![Expr::Stack(vec![Expr::Num(100.0), Expr::Num(200.0)])]
                }
            })]
        );
        assert!(matches!(
            parse_src("{x 1} = 1"),
            Err(ParseError::InvalidAssignment(Expr::Num(..)))
        ));

        Ok(())
    }

    #[test]
    fn parse_lex_error() {
        match parse_src("1.2.3") {
            Err(ParseError::LexError(err)) => {
                assert_eq!(err, LexError::InvalidNumber("1.2.3".into()));
            }
            other => panic!("Expected unrecognized character, got {other:?}"),
        }
    }

    fn assert_expected_ident(src: &str, expected: &Expr) {
        match parse_src(src) {
            Err(ParseError::ExpectedIdent(token)) => assert_eq!(token, *expected),
            other => panic!("Expected ExpectedIdent({expected:?}), got {other:?}"),
        }
    }

    #[test]
    fn parse_unexpected_token() {
        assert_expected_ident("x = |1| a", &Expr::Num(1.0));
        assert_expected_ident("1 = 2", &Expr::Num(1.0));
    }

    fn assert_expected_primary(src: &str, expected: &Token) {
        match parse_src(src) {
            Err(ParseError::ExpectedPrimary(token)) => assert_eq!(token, *expected),
            other => panic!("Expected ExpectedPrimary({expected:?}), got {other:?}"),
        }
    }

    #[test]
    fn parse_expected_primary() {
        assert_expected_primary("+ 2 3", &Token::Plus);
        assert_expected_primary("= 5", &Token::Assignment);
    }

    #[test]
    fn parse_invalid_assignment_target() {
        match parse_src("a 1 = 1") {
            Err(ParseError::InvalidAssignment(expr)) => assert_eq!(expr, Expr::Num(1.0)),
            other => panic!("Expected InvalidAssignment(1.0), got {other:?}"),
        }
    }

    #[test]
    fn parse_empty() -> Result<(), ParseError> {
        assert_eq!(parse_src("")?, vec![]);
        assert_eq!(parse_src("\n")?, vec![]);

        Ok(())
    }

    #[test]
    fn parse_multiple_statements() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("a = 1\nb = 2\nsin a")?,
            vec![
                Ast::Assign(Assign {
                    names: vec!["a".into()],
                    rhs: Expr::Num(1.0)
                }),
                Ast::Assign(Assign {
                    names: vec!["b".into()],
                    rhs: Expr::Num(2.0)
                }),
                Ast::Expr(Expr::Call {
                    callee: Expr::Ident("sin".into()).into(),
                    args: vec![Expr::Ident("a".into())],
                }),
            ]
        );

        Ok(())
    }

    #[test]
    fn parse_cycle() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("a = sin ~a")?,
            vec![Ast::Assign(Assign {
                names: vec!["a".into()],
                rhs: Expr::Call {
                    callee: Expr::Ident("sin".into()).into(),
                    args: vec![Expr::Tap {
                        name: "a".into(),
                        index: Expr::Num(1.0).into()
                    }]
                }
            })]
        );

        Ok(())
    }

    #[test]
    fn parse_tap() -> Result<(), ParseError> {
        // the index binds to the ref, not as an argument
        assert_eq!(
            parse_src("sin ~a[n + 1] 2")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Ident("sin".into()).into(),
                args: vec![
                    Expr::Tap {
                        name: "a".into(),
                        index: Expr::Binary(
                            Expr::Ident("n".into()).into(),
                            BinaryOp::Add,
                            Expr::Num(1.0).into()
                        )
                        .into()
                    },
                    Expr::Num(2.0)
                ]
            })]
        );
        assert!(matches!(
            parse_src("~a[1"),
            Err(ParseError::UnclosedIndex(Token::Eof))
        ));

        Ok(())
    }

    #[test]
    fn parse_comment() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("x = 1\n// TODO: fix\ny = 2")?,
            vec![
                Ast::Assign(Assign {
                    names: vec!["x".into()],
                    rhs: Expr::Num(1.0)
                }),
                Ast::Assign(Assign {
                    names: vec!["y".into()],
                    rhs: Expr::Num(2.0)
                })
            ]
        );

        Ok(())
    }

    #[test]
    fn parse_seq() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("<100 200>")?,
            vec![Ast::Expr(Expr::Seq(vec![
                Expr::Num(100.0),
                Expr::Num(200.0)
            ]))]
        );
        assert_eq!(
            parse_src("x = 100\n<x 200>")?,
            vec![
                Ast::Assign(Assign {
                    names: vec!["x".into()],
                    rhs: Expr::Num(100.0)
                }),
                Ast::Expr(Expr::Seq(vec![Expr::Ident("x".into()), Expr::Num(200.0)]))
            ]
        );

        Ok(())
    }

    #[test]
    fn parse_stack() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("{100 200}")?,
            vec![Ast::Expr(Expr::Stack(vec![
                Expr::Num(100.0),
                Expr::Num(200.0)
            ]))]
        );
        assert_eq!(
            parse_src("x = 100\n{x 200}")?,
            vec![
                Ast::Assign(Assign {
                    names: vec!["x".into()],
                    rhs: Expr::Num(100.0)
                }),
                Ast::Expr(Expr::Stack(vec![Expr::Ident("x".into()), Expr::Num(200.0)]))
            ]
        );

        Ok(())
    }

    #[test]
    fn parse_seq_call() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("ramp 1 -> <100>")?,
            vec![Ast::Expr(Expr::Call {
                callee: Expr::Seq(vec![Expr::Num(100.0)]).into(),
                args: vec![Expr::Call {
                    callee: Expr::Ident("ramp".into()).into(),
                    args: vec![Expr::Num(1.0)]
                }]
            })]
        );
        assert_eq!(
            parse_src("x = 100\n<x 200>")?,
            vec![
                Ast::Assign(Assign {
                    names: vec!["x".into()],
                    rhs: Expr::Num(100.0)
                }),
                Ast::Expr(Expr::Seq(vec![Expr::Ident("x".into()), Expr::Num(200.0)]))
            ]
        );

        Ok(())
    }

    #[test]
    fn parse_nested_seq() -> Result<(), ParseError> {
        assert_eq!(
            parse_src("<100 <100 200>>")?,
            vec![Ast::Expr(Expr::Seq(vec![
                Expr::Num(100.0),
                Expr::Seq(vec![Expr::Num(100.0), Expr::Num(200.0)])
            ]))]
        );

        Ok(())
    }
}
