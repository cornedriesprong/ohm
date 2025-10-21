use crate::nodes::{
    BufReaderNode, BufRefNode, DelayNode, DivideNode, EqualNode, FunDSPNode, GainNode, GreaterNode,
    LFONode, LessNode, MixNode, Node, PowerNode, RampNode, SampleAndHoldNode, SeqNode,
    SubtractNode, WrapNode,
};
use crate::utils::get_audio_frames;
use fundsp::hacker32::*;
use std::{any::Any, collections::HashMap};

#[derive(Clone, Debug)]
pub(crate) enum Token {
    Number(f32),
    Identifier(String),
    String(String),
    Plus,
    Minus,
    Multiply,
    Divide,
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
                tokens.push(Token::Plus);
                chars.next();
            }
            '-' => {
                tokens.push(Token::Minus);
                chars.next();
            }
            '*' => {
                tokens.push(Token::Multiply);
                chars.next();
            }
            '/' => {
                tokens.push(Token::Divide);
                chars.next();
            }
            '%' => {
                tokens.push(Token::Modulo);
                chars.next();
            }
            '^' => {
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
            '"' => {
                chars.next();
                let mut str_content = String::new();
                let mut terminated = false;

                while let Some(&ch) = chars.peek() {
                    if ch == '"' {
                        chars.next();
                        terminated = true;
                        break;
                    }
                    str_content.push(ch);
                    chars.next();
                }

                if !terminated {
                    panic!("Unterminated string literal");
                }

                tokens.push(Token::String(str_content));
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
            while matches!(self.peek(), Token::Newline) {
                self.consume();
            }

            if matches!(self.peek(), Token::Eof) {
                break;
            }

            last_expr = self.parse_statement();

            while matches!(self.peek(), Token::Newline) {
                self.consume();
            }
        }

        last_expr
    }

    fn parse_statement(&mut self) -> Option<Box<dyn Node>> {
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

            self.pos = prev_pos;
        }

        self.parse_expr(0)
    }

    fn parse_expr(&mut self, min_prec: u8) -> Option<Box<dyn Node>> {
        let mut lhs = self.parse_primary()?;

        while let Some(op_prec) = self.get_precedence(self.peek()) {
            if op_prec < min_prec {
                break;
            }

            let op = self.consume();
            let rhs = self.parse_expr(op_prec + 1)?;

            lhs = match op {
                Token::Plus => Box::new(MixNode::new(vec![lhs, rhs])),
                Token::Minus => Box::new(SubtractNode::new(vec![lhs, rhs])),
                Token::Multiply => Box::new(GainNode::new(vec![lhs, rhs])),
                Token::Divide => Box::new(DivideNode::new(vec![lhs, rhs])),
                Token::Modulo => Box::new(WrapNode::new(vec![lhs, rhs])),
                Token::Power => Box::new(PowerNode::new(vec![lhs, rhs])),
                Token::Greater => Box::new(GreaterNode::new(vec![lhs, rhs])),
                Token::Less => Box::new(LessNode::new(vec![lhs, rhs])),
                Token::Equal => Box::new(EqualNode::new(vec![lhs, rhs])),
                _ => lhs,
            };
        }

        Some(lhs)
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

    fn parse_str(&mut self) -> Option<String> {
        match self.consume() {
            Token::String(s) => Some(s),
            _ => None,
        }
    }

    fn parse_node(&mut self, name: &str) -> Option<Box<dyn Node>> {
        match name {
            "ramp" => {
                let freq = self.parse_primary().unwrap_or_else(|| constant_node(1.0));
                Some(Box::new(RampNode::new(vec![freq], self.sample_rate)))
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
            "seq" => {
                let mut args = Vec::new();
                while let Some(arg) = self.parse_primary() {
                    args.push(arg);
                }
                Some(Box::new(SeqNode::new(args)))
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
                let input = self.parse_primary()?;
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(500.0));
                let resonance = self.parse_primary().unwrap_or_else(|| constant_node(0.707));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(lowpass()),
                )))
            }
            "bp" => {
                let input = self.parse_primary()?;
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(500.0));
                let resonance = self.parse_primary().unwrap_or_else(|| constant_node(0.707));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(bandpass()),
                )))
            }
            "hp" => {
                let input = self.parse_primary()?;
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(500.0));
                let resonance = self.parse_primary().unwrap_or_else(|| constant_node(0.707));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(highpass()),
                )))
            }
            "moog" => {
                let input = self.parse_primary()?;
                let cutoff = self.parse_primary().unwrap_or_else(|| constant_node(500.0));
                let resonance = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                Some(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(moog()),
                )))
            }
            "delay" => {
                let input = self.parse_primary()?;
                let delay_time = self
                    .parse_primary()
                    .unwrap_or_else(|| constant_node(1000.0));
                Some(Box::new(DelayNode::new(vec![input, delay_time])))
            }
            "reverb" => {
                let input = self.parse_primary()?;
                Some(Box::new(FunDSPNode::stereo(
                    vec![input],
                    Box::new(reverb2_stereo(10.0, 2.0, 0.9, 1.0, lowpole_hz(18000.0))),
                )))
            }
            "file" => {
                let name = self.parse_str()?;
                let frames = get_audio_frames(&name);
                Some(Box::new(BufRefNode::new(name, frames)))
            }
            "play" => {
                let buf_ref = self.parse_primary()?;
                if let Some(buf_ref) = (buf_ref.as_ref() as &dyn Any).downcast_ref::<BufRefNode>() {
                    let phase = self.parse_primary().unwrap_or_else(|| constant_node(0.0));
                    Some(Box::new(BufReaderNode::new(buf_ref.clone(), vec![phase])))
                } else {
                    panic!("Expected a buffer reference for 'play'");
                }
            }
            _ => None,
        }
    }

    fn get_precedence(&self, token: &Token) -> Option<u8> {
        match token {
            Token::Power => Some(1),
            Token::Multiply | Token::Divide | Token::Modulo => Some(2),
            Token::Plus | Token::Minus => Some(3),
            Token::Greater | Token::Less | Token::Equal => Some(4),
            _ => None,
        }
    }
}
