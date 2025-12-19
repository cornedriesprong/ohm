use crate::tokenizer::{tokenize, Token};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, PartialEq, Debug)]
pub enum Expr {
    Number(f32),
    Ref(String),
    FbRef(String),
    Assign {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    Func {
        params: Vec<String>,
        body: Box<Expr>,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Box<Expr>>,
    },
}

pub(crate) struct Parser {
    tokens: VecDeque<Token>,
    pos: usize,
    env: HashMap<String, Expr>,
}

impl Parser {
    pub(crate) fn new(src: String) -> Self {
        Self {
            tokens: tokenize(src),
            pos: 0,
            env: HashMap::new(),
        }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn consume(&mut self) -> Token {
        let tok = self.peek().clone();
        self.pos += 1;
        tok
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline) {
            self.consume();
        }
    }

    pub(crate) fn parse(mut self) -> Option<Expr> {
        self.skip_newlines();

        if matches!(self.peek(), Token::Eof) {
            return None;
        }

        // try to parse assignment
        if let Token::Identifier(name) = self.peek() {
            let name = name.clone();
            let saved_pos = self.pos;
            self.consume();

            if matches!(self.peek(), Token::Assign) {
                self.consume();
                if let Some(value) = self.parse_expr(0) {
                    let body = self.parse().unwrap_or(Expr::Ref(name.clone()));
                    return Some(Expr::Assign {
                        name,
                        value: Box::new(value),
                        body: Box::new(body),
                    });
                }
            }

            self.pos = saved_pos;
        }

        // not an assignment, just parse and return one expression
        self.parse_expr(0)
    }

    fn parse_expr(&mut self, min_prec: u8) -> Option<Expr> {
        let mut lhs = self.parse_primary()?;

        // handle function application
        let mut args = Vec::new();
        while matches!(
            self.peek(),
            Token::Number(_)
                | Token::Identifier(_)
                | Token::LParen
                | Token::Pipe
                | Token::Minus
                | Token::Tilde
        ) && self.op_precedence(self.peek()).is_none()
        {
            args.push(Box::new(self.parse_primary()?));
        }
        if !args.is_empty() {
            lhs = Expr::Call {
                func: Box::new(lhs),
                args,
            };
        }

        while let Some(op_prec) = self.op_precedence(self.peek()) {
            if op_prec < min_prec {
                break;
            }

            let op = self.consume();

            lhs = match op {
                Token::Chain => {
                    let rhs = self.parse_expr(op_prec + 1)?;
                    Expr::Call {
                        func: Box::new(rhs),
                        args: vec![Box::new(lhs)],
                    }
                }
                _ => {
                    let rhs = self.parse_expr(op_prec + 1)?;
                    let name = self.op_name(&op)?;
                    Expr::Call {
                        func: Box::new(Expr::Ref(name.to_string())),
                        args: vec![Box::new(lhs), Box::new(rhs)],
                    }
                }
            };
        }

        Some(lhs)
    }

    fn parse_func(&mut self) -> Option<Expr> {
        let mut params = Vec::new();

        while let Token::Identifier(name) = self.peek() {
            params.push(name.clone());
            self.consume();
        }

        if !matches!(self.peek(), Token::Pipe) {
            panic!("Expected | after function parameters");
        }
        self.consume();

        let body = self.parse_expr(0)?;

        Some(Expr::Func {
            params,
            body: Box::new(body),
        })
    }

    fn parse_primary(&mut self) -> Option<Expr> {
        let pos = self.pos;

        let result = match self.consume() {
            Token::Number(num) => Some(Expr::Number(num)),
            Token::Minus => {
                // negative number
                if let Token::Number(num) = self.peek() {
                    let num = *num;
                    self.consume();
                    Some(Expr::Number(-num))
                } else {
                    None
                }
            }
            Token::Tilde => {
                if let Token::Identifier(name) = self.peek() {
                    let name = name.clone();
                    self.consume();
                    Some(Expr::FbRef(name))
                } else {
                    panic!("Expected identifier after ~")
                }
            }
            Token::Identifier(name) => {
                if let Some(node_id) = self.env.get(&name) {
                    Some(node_id.clone())
                } else {
                    Some(Expr::Ref(name))
                }
            }
            Token::LParen => {
                let expr = self.parse_expr(0)?;
                match self.consume() {
                    Token::RParen => Some(expr),
                    _ => panic!("Expected closing parenthesis"),
                }
            }
            Token::Pipe => {
                if let Some(expr) = self.parse_func() {
                    Some(expr)
                } else {
                    None
                }
            }
            _ => None,
        };

        // backtrack if we failed to parse
        if result.is_none() {
            self.pos = pos;
        }

        result
    }

    fn op_precedence(&self, token: &Token) -> Option<u8> {
        match token {
            Token::Chain => Some(0),
            Token::Power => Some(1),
            Token::Multiply | Token::Divide | Token::Modulo => Some(2),
            Token::Plus | Token::Minus => Some(3),
            Token::Greater | Token::Less | Token::Equal => Some(4),
            _ => None,
        }
    }

    fn op_name(&self, token: &Token) -> Option<&'static str> {
        match token {
            Token::Plus => Some("sum"),
            Token::Minus => Some("diff"),
            Token::Multiply => Some("mul"),
            Token::Divide => Some("divide"),
            Token::Modulo => Some("wrap"),
            Token::Power => Some("power"),
            Token::Greater => Some("greater"),
            Token::Less => Some("less"),
            Token::Equal => Some("equal"),
            _ => None,
        }
    }
}
