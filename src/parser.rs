use crate::nodes::{
    DelayNode, EqualNode, FunDSPNode, GainNode, GreaterNode, LFONode, LessNode, MixNode, Node,
    PowerNode, SampleAndHoldNode, WrapNode,
};
use fundsp::hacker32::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(crate) enum Token {
    Number(f32),
    Identifier(String),
    Add,
    Multiply,
    Modulo,
    Power,
    Greater,
    Less,
    Equal,
    LParen,
    RParen,
    Newline,
    Eof,
}

fn constant_node(value: f32) -> Box<dyn Node> {
    Box::new(FunDSPNode::mono(vec![], Box::new(dc(value))))
}

pub(crate) fn tokenize(str: String) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = str.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\r' => {
                chars.next();
            }
            '\n' => {
                tokens.push(Token::Newline);
                chars.next();
            }
            '+' => {
                tokens.push(Token::Add);
                chars.next();
            }
            '*' => {
                tokens.push(Token::Multiply);
                chars.next();
            }
            '%' => {
                tokens.push(Token::Modulo);
                chars.next();
            }
            '^' => {
                println!("Power operator found");
                tokens.push(Token::Power);
                chars.next();
            }
            '>' => {
                tokens.push(Token::Greater);
                chars.next();
            }
            '<' => {
                tokens.push(Token::Less);
                chars.next();
            }
            '=' => {
                tokens.push(Token::Equal);
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut num_str = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_ascii_digit() || ch == '.' {
                        num_str.push(ch);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if let Ok(num) = num_str.parse::<f32>() {
                    tokens.push(Token::Number(num));
                } else {
                    panic!("Invalid number: {}", num_str);
                }
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut ident = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || ch == '_' {
                        ident.push(ch);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Identifier(ident));
            }
            _ => panic!("Unrecognized character: {}", ch),
        }
    }

    tokens.push(Token::Eof);
    tokens
}

pub(crate) struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    env: HashMap<String, Box<dyn Node>>,
    sample_rate: u32,
}

impl Parser {
    pub(crate) fn new(tokens: Vec<Token>, sample_rate: u32) -> Self {
        return Self {
            tokens,
            pos: 0,
            env: HashMap::new(),
            sample_rate,
        };
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn consume(&mut self) -> Token {
        let tok = self.peek().clone();
        self.pos += 1;
        tok
    }

    pub(crate) fn parse(&mut self) -> Option<Box<dyn Node>> {
        let mut last_expr = None;

        loop {
            // Skip any leading newlines
            while matches!(self.peek(), Token::Newline) {
                self.consume();
            }

            // Check for end of input
            if matches!(self.peek(), Token::Eof) {
                break;
            }

            // Try to parse a statement
            last_expr = self.parse_statement();

            // Consume trailing newlines or expect EOF
            while matches!(self.peek(), Token::Newline) {
                self.consume();
            }
        }

        last_expr
    }

    fn parse_statement(&mut self) -> Option<Box<dyn Node>> {
        // Try to parse assignment: identifier = expr
        if let Token::Identifier(name) = self.peek() {
            let name = name.clone();
            let prev_pos = self.pos;
            self.consume();

            if matches!(self.peek(), Token::Equal) {
                self.consume();
                if let Some(expr) = self.parse_expr(0) {
                    self.env.insert(name, expr.clone());
                    return Some(expr);
                }
            }

            // Not an assignment, backtrack
            self.pos = prev_pos;
        }

        // Otherwise, parse as expression
        self.parse_expr(0)
    }

    fn parse_expr(&mut self, min_prec: u8) -> Option<Box<dyn Node>> {
        let mut left = self.parse_primary()?;

        while let Some(op_prec) = self.get_precedence(self.peek()) {
            if op_prec < min_prec {
                break;
            }

            let op = self.consume();
            let right = self.parse_expr(op_prec + 1)?;

            left = match op {
                Token::Add => Box::new(MixNode::new(vec![left, right])),
                Token::Multiply => Box::new(GainNode::new(vec![left, right])),
                Token::Modulo => Box::new(WrapNode::new(vec![left, right])),
                Token::Power => Box::new(PowerNode::new(vec![left, right])),
                Token::Greater => Box::new(GreaterNode::new(vec![left, right])),
                Token::Less => Box::new(LessNode::new(vec![left, right])),
                Token::Equal => Box::new(EqualNode::new(vec![left, right])),
                _ => left,
            };
        }

        Some(left)
    }

    fn parse_primary(&mut self) -> Option<Box<dyn Node>> {
        match self.consume() {
            Token::Number(num) => Some(Box::new(FunDSPNode::mono(vec![], Box::new(dc(num))))),
            Token::Identifier(name) => {
                if let Some(op) = self.parse_node(&name) {
                    Some(op)
                } else if let Some(node) = self.env.get(&name) {
                    Some(node.clone())
                } else {
                    panic!("Undefined variable: {}", name);
                }
            }
            Token::LParen => {
                let expr = self.parse_expr(0)?;
                match self.consume() {
                    Token::RParen => Some(expr),
                    _ => panic!("Expected closing parenthesis"),
                }
            }
            _ => None,
        }
    }

    fn parse_node(&mut self, name: &str) -> Option<Box<dyn Node>> {
        match name {
            "ramp" => {
                let freq = self.parse_primary().unwrap_or_else(|| constant_node(1.0));
                Some(Box::new(FunDSPNode::mono(vec![freq], Box::new(saw()))))
            }
            "sin" => {
                let freq = self.parse_primary().unwrap_or_else(|| constant_node(100.0));
                Some(Box::new(FunDSPNode::mono(vec![freq], Box::new(sine()))))
            }
            "saw" => {
                let freq = self.parse_primary().unwrap_or_else(|| constant_node(100.0));
                Some(Box::new(FunDSPNode::mono(vec![freq], Box::new(saw()))))
            }
            "sqr" => {
                let freq = self.parse_primary().unwrap_or_else(|| constant_node(100.0));
                Some(Box::new(FunDSPNode::mono(vec![freq], Box::new(square()))))
            }
            "tri" => {
                let freq = self.parse_primary().unwrap_or_else(|| constant_node(100.0));
                Some(Box::new(FunDSPNode::mono(vec![freq], Box::new(triangle()))))
            }
            "noise" => Some(Box::new(FunDSPNode::mono(vec![], Box::new(noise())))),
            "clip" => {
                let input = self
                    .parse_primary()
                    .unwrap_or_else(|| Box::new(FunDSPNode::mono(vec![], Box::new(dc(0.0)))));
                Some(Box::new(FunDSPNode::mono(vec![input], Box::new(clip()))))
            }
            "lfo" => {
                let freq = self.parse_primary().unwrap_or_else(|| constant_node(1.0));
                Some(Box::new(LFONode::new(vec![freq], self.sample_rate)))
            }
            "sh" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                let trig = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                Some(Box::new(SampleAndHoldNode::new(
                    vec![input, trig],
                    self.sample_rate,
                )))
            }
            "pan" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                let pan = self.parse_primary().unwrap_or_else(|| constant_node(0.5));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, pan],
                    Box::new(panner()),
                )))
            }
            "onepole" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(20.0));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff],
                    Box::new(lowpole()),
                )))
            }
            "lp" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(500.0));
                let resonance = self.parse_primary().unwrap_or_else(|| constant_node(0.707));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(lowpass()),
                )))
            }
            "bp" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(500.0));
                let resonance = self.parse_primary().unwrap_or_else(|| constant_node(0.707));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(bandpass()),
                )))
            }
            "hp" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(500.0));
                let resonance = self.parse_primary().unwrap_or_else(|| constant_node(0.707));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(highpass()),
                )))
            }
            "moog" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(500.0));
                let resonance = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(moog()),
                )))
            }
            "delay" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                let delay_time = self
                    .parse_primary()
                    .unwrap_or_else(|| constant_node(1000.0));
                Some(Box::new(DelayNode::new(vec![input, delay_time])))
            }
            "reverb" => {
                let input = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                Some(Box::new(FunDSPNode::stereo(
                    vec![input],
                    Box::new(reverb2_stereo(10.0, 2.0, 0.9, 1.0, lowpole_hz(18000.0))),
                )))
            }
            _ => None,
        }
    }

    fn get_precedence(&self, token: &Token) -> Option<u8> {
        match token {
            Token::Power => Some(1),
            Token::Multiply | Token::Modulo => Some(2),
            Token::Add => Some(3),
            Token::Greater | Token::Less | Token::Equal => Some(4),
            _ => None,
        }
    }
}
