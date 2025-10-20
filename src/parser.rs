use crate::audio_graph::Graph;
use crate::nodes::{
    BufReaderNode, BufTapNode, BufWriterNode, DelayNode, FunDSPNode, LFONode, NodeKind,
    SampleAndHoldNode, SeqNode,
};
use crate::op::Op;
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
    env: HashMap<String, Op>,
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

    pub(crate) fn parse(&mut self) -> Option<Op> {
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

    fn parse_statement(&mut self) -> Option<Op> {
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

    fn parse_expr(&mut self, min_prec: u8) -> Option<Op> {
        let mut left = self.parse_primary()?;

        while let Some(op_prec) = self.get_precedence(self.peek()) {
            if op_prec < min_prec {
                break;
            }

            let op = self.consume();
            let right = self.parse_expr(op_prec + 1)?;

            left = match op {
                Token::Add => Op::Mix(Box::new(left), Box::new(right)),
                Token::Multiply => Op::Gain(Box::new(left), Box::new(right)),
                Token::Modulo => Op::Wrap(Box::new(left), Box::new(right)),
                Token::Power => Op::Power(Box::new(left), Box::new(right)),
                Token::Greater => Op::Greater(Box::new(left), Box::new(right)),
                Token::Less => Op::Less(Box::new(left), Box::new(right)),
                Token::Equal => Op::Equal(Box::new(left), Box::new(right)),
                _ => unreachable!(),
            };
        }

        Some(left)
    }

    fn parse_primary(&mut self) -> Option<Op> {
        match self.consume() {
            Token::Number(num) => Some(Op::Constant(num)),
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

    fn parse_node(&mut self, name: &str) -> Option<Op> {
        match name {
            "ramp" => {
                let freq = self.parse_primary().unwrap_or(Op::Constant(1.0));
                Some(Op::Node {
                    kind: NodeKind::Ramp,
                    inputs: vec![freq],
                    node: Box::new(FunDSPNode::mono(Box::new(sine()))),
                })
            }
            "sin" => {
                let freq = self.parse_primary().unwrap_or(Op::Constant(100.0));
                Some(Op::Node {
                    kind: NodeKind::Sin,
                    inputs: vec![freq],
                    node: Box::new(FunDSPNode::mono(Box::new(sine()))),
                })
            }
            "saw" => {
                let freq = self.parse_primary().unwrap_or(Op::Constant(100.0));
                Some(Op::Node {
                    kind: NodeKind::Saw,
                    inputs: vec![freq],
                    node: Box::new(FunDSPNode::mono(Box::new(saw()))),
                })
            }
            "sqr" => {
                let freq = self.parse_primary().unwrap_or(Op::Constant(100.0));
                Some(Op::Node {
                    kind: NodeKind::Sqr,
                    inputs: vec![freq],
                    node: Box::new(FunDSPNode::mono(Box::new(square()))),
                })
            }
            "tri" => {
                let freq = self.parse_primary().unwrap_or(Op::Constant(100.0));
                Some(Op::Node {
                    kind: NodeKind::Tri,
                    inputs: vec![freq],
                    node: Box::new(FunDSPNode::mono(Box::new(triangle()))),
                })
            }
            "noise" => Some(Op::Node {
                kind: NodeKind::Noise,
                inputs: vec![],
                node: Box::new(FunDSPNode::mono(Box::new(noise()))),
            }),
            "clip" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                Some(Op::Node {
                    kind: NodeKind::Clip,
                    inputs: vec![input],
                    node: Box::new(FunDSPNode::mono(Box::new(clip()))),
                })
            }
            "lfo" => {
                let freq = self.parse_primary().unwrap_or(Op::Constant(1.0));
                Some(Op::Node {
                    kind: NodeKind::Lfo,
                    inputs: vec![freq],
                    node: Box::new(LFONode::new(self.sample_rate)),
                })
            }
            "sh" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                let trig = self.parse_primary().unwrap_or(Op::Constant(0.0));
                Some(Op::Node {
                    kind: NodeKind::SampleAndHold,
                    inputs: vec![input, trig],
                    node: Box::new(SampleAndHoldNode::new()),
                })
            }
            "log" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                Some(Op::Node {
                    kind: NodeKind::Log,
                    inputs: vec![input],
                    node: Box::new(FunDSPNode::mono(Box::new(sink()))),
                })
            }
            "pan" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                let pan = self.parse_primary().unwrap_or(Op::Constant(0.5));
                Some(Op::Node {
                    kind: NodeKind::Pan,
                    inputs: vec![input, pan],
                    node: Box::new(FunDSPNode::mono(Box::new(panner()))),
                })
            }
            "onepole" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                let cutoff = self.parse_primary().unwrap_or(Op::Constant(20.0));
                Some(Op::Node {
                    kind: NodeKind::Onepole,
                    inputs: vec![input, cutoff],
                    node: Box::new(FunDSPNode::mono(Box::new(lowpass()))),
                })
            }
            "lp" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                let cutoff = self.parse_primary().unwrap_or(Op::Constant(500.0));
                let resonance = self.parse_primary().unwrap_or(Op::Constant(0.707));
                Some(Op::Node {
                    kind: NodeKind::Svf,
                    inputs: vec![input, cutoff, resonance],
                    node: Box::new(FunDSPNode::mono(Box::new(lowpass()))),
                })
            }
            "bp" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                let cutoff = self.parse_primary().unwrap_or(Op::Constant(500.0));
                let resonance = self.parse_primary().unwrap_or(Op::Constant(0.707));
                Some(Op::Node {
                    kind: NodeKind::Svf,
                    inputs: vec![input, cutoff, resonance],
                    node: Box::new(FunDSPNode::mono(Box::new(bandpass()))),
                })
            }
            "hp" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                let cutoff = self.parse_primary().unwrap_or(Op::Constant(500.0));
                let resonance = self.parse_primary().unwrap_or(Op::Constant(0.707));
                Some(Op::Node {
                    kind: NodeKind::Svf,
                    inputs: vec![input, cutoff, resonance],
                    node: Box::new(FunDSPNode::mono(Box::new(highpass()))),
                })
            }
            "moog" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                let cutoff = self.parse_primary().unwrap_or(Op::Constant(500.0));
                let resonance = self.parse_primary().unwrap_or(Op::Constant(0.0));
                Some(Op::Node {
                    kind: NodeKind::Moog,
                    inputs: vec![input, cutoff, resonance],
                    node: Box::new(FunDSPNode::mono(Box::new(moog()))),
                })
            }
            "delay" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                Some(Op::Node {
                    kind: NodeKind::Delay,
                    inputs: vec![input],
                    node: Box::new(DelayNode::new()),
                })
            }
            "reverb" => {
                let input = self.parse_primary().unwrap_or(Op::Constant(0.0));
                Some(Op::Node {
                    kind: NodeKind::Reverb,
                    inputs: vec![input],
                    node: Box::new(FunDSPNode::stereo(Box::new(reverb2_stereo(
                        10.0,
                        2.0,
                        0.9,
                        1.0,
                        lowpole_hz(18000.0),
                    )))),
                })
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
