use crate::container::Arena;
use crate::nodes::{
    BufReaderNode, BufRefNode, BufTapNode, BufWriterNode, DelayNode, DivideNode, EqualNode,
    FunDSPNode, GainNode, GreaterNode, LFONode, LessNode, MixNode, PowerNode, RampNode,
    SampleAndHoldNode, SeqNode, SubtractNode, WrapNode,
};
use crate::utils::get_audio_frames;
use fundsp::hacker32::*;
use std::collections::{HashMap, VecDeque};

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

fn constant_node(arena: &mut Arena, value: f32) -> usize {
    arena.alloc(Box::new(FunDSPNode::mono(vec![], Box::new(dc(value)))))
}

fn tokenize(str: String) -> VecDeque<Token> {
    let mut tokens = VecDeque::new();
    let mut chars = str.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\r' => {
                chars.next();
            }
            '\n' => {
                tokens.push_back(Token::Newline);
                chars.next();
            }
            '+' => {
                tokens.push_back(Token::Plus);
                chars.next();
            }
            '-' => {
                tokens.push_back(Token::Minus);
                chars.next();
            }
            '*' => {
                tokens.push_back(Token::Multiply);
                chars.next();
            }
            '/' => {
                tokens.push_back(Token::Divide);
                chars.next();
            }
            '%' => {
                tokens.push_back(Token::Modulo);
                chars.next();
            }
            '^' => {
                tokens.push_back(Token::Power);
                chars.next();
            }
            '>' => {
                tokens.push_back(Token::Greater);
                chars.next();
            }
            '<' => {
                tokens.push_back(Token::Less);
                chars.next();
            }
            '=' => {
                tokens.push_back(Token::Equal);
                chars.next();
            }
            '(' => {
                tokens.push_back(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push_back(Token::RParen);
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
                    tokens.push_back(Token::Number(num));
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

                tokens.push_back(Token::String(str_content));
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
                tokens.push_back(Token::Identifier(ident));
            }
            _ => panic!("Unrecognized character: {}", ch),
        }
    }

    tokens.push_back(Token::Eof);
    tokens
}

pub(crate) struct Parser {
    tokens: VecDeque<Token>,
    pos: usize,
    env: HashMap<String, usize>,
    sample_rate: u32,
}

impl Parser {
    pub(crate) fn new(src: String, sample_rate: u32) -> Self {
        return Self {
            tokens: tokenize(src),
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

    pub(crate) fn parse(&mut self, arena: &mut Arena) -> Option<usize> {
        let mut last_expr = None;

        loop {
            while matches!(self.peek(), Token::Newline) {
                self.consume();
            }

            if matches!(self.peek(), Token::Eof) {
                break;
            }

            last_expr = self.parse_statement(arena);

            while matches!(self.peek(), Token::Newline) {
                self.consume();
            }
        }

        last_expr
    }

    fn parse_statement(&mut self, arena: &mut Arena) -> Option<usize> {
        if let Token::Identifier(name) = self.peek() {
            let name = name.clone();
            let prev_pos = self.pos;
            self.consume();

            if matches!(self.peek(), Token::Equal) {
                self.consume();
                if let Some(expr) = self.parse_expr(arena, 0) {
                    self.env.insert(name, expr);
                    return Some(expr);
                }
            }

            self.pos = prev_pos;
        }

        self.parse_expr(arena, 0)
    }

    fn parse_expr(&mut self, arena: &mut Arena, min_prec: u8) -> Option<usize> {
        let mut lhs = self.parse_primary(arena)?;

        while let Some(op_prec) = self.get_precedence(self.peek()) {
            if op_prec < min_prec {
                break;
            }

            let op = self.consume();
            let rhs = self.parse_expr(arena, op_prec + 1)?;

            lhs = match op {
                Token::Plus => arena.alloc(Box::new(MixNode::new(vec![lhs, rhs]))),
                Token::Minus => arena.alloc(Box::new(SubtractNode::new(vec![lhs, rhs]))),
                Token::Multiply => arena.alloc(Box::new(GainNode::new(vec![lhs, rhs]))),
                Token::Divide => arena.alloc(Box::new(DivideNode::new(vec![lhs, rhs]))),
                Token::Modulo => arena.alloc(Box::new(WrapNode::new(vec![lhs, rhs]))),
                Token::Power => arena.alloc(Box::new(PowerNode::new(vec![lhs, rhs]))),
                Token::Greater => arena.alloc(Box::new(GreaterNode::new(vec![lhs, rhs]))),
                Token::Less => arena.alloc(Box::new(LessNode::new(vec![lhs, rhs]))),
                Token::Equal => arena.alloc(Box::new(EqualNode::new(vec![lhs, rhs]))),
                _ => lhs,
            };
        }

        Some(lhs)
    }

    fn parse_primary(&mut self, arena: &mut Arena) -> Option<usize> {
        match self.consume() {
            Token::Number(num) => {
                Some(arena.alloc(Box::new(FunDSPNode::mono(vec![], Box::new(dc(num))))))
            }
            Token::Identifier(name) => {
                if let Some(op) = self.parse_node(arena, &name) {
                    Some(op)
                } else if let Some(&node_id) = self.env.get(&name) {
                    Some(node_id)
                } else {
                    panic!("Undefined variable: {}", name);
                }
            }
            Token::LParen => {
                let expr = self.parse_expr(arena, 0)?;
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

    fn parse_node(&mut self, arena: &mut Arena, name: &str) -> Option<usize> {
        match name {
            "ramp" => {
                let freq = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 1.0));
                Some(arena.alloc(Box::new(RampNode::new(vec![freq], self.sample_rate))))
            }
            "sin" => {
                let freq = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 100.0));
                Some(arena.alloc(Box::new(FunDSPNode::mono(vec![freq], Box::new(sine())))))
            }
            "saw" => {
                let freq = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 100.0));
                Some(arena.alloc(Box::new(FunDSPNode::mono(vec![freq], Box::new(saw())))))
            }
            "sqr" => {
                let freq = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 100.0));
                Some(arena.alloc(Box::new(FunDSPNode::mono(vec![freq], Box::new(square())))))
            }
            "tri" => {
                let freq = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 100.0));
                Some(arena.alloc(Box::new(FunDSPNode::mono(vec![freq], Box::new(triangle())))))
            }
            "noise" => Some(arena.alloc(Box::new(FunDSPNode::mono(vec![], Box::new(noise()))))),
            "clip" => {
                let input = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                Some(arena.alloc(Box::new(FunDSPNode::mono(vec![input], Box::new(clip())))))
            }
            "lfo" => {
                let freq = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 1.0));
                Some(arena.alloc(Box::new(LFONode::new(vec![freq], self.sample_rate))))
            }
            "sh" => {
                let input = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                let trig = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                Some(arena.alloc(Box::new(SampleAndHoldNode::new(
                    vec![input, trig],
                    self.sample_rate,
                ))))
            }
            "pan" => {
                let input = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                let pan = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.5));
                Some(arena.alloc(Box::new(FunDSPNode::mono(
                    vec![input, pan],
                    Box::new(panner()),
                ))))
            }
            "seq" => {
                let mut args = Vec::new();
                while let Some(arg) = self.parse_primary(arena) {
                    args.push(arg);
                }
                Some(arena.alloc(Box::new(SeqNode::new(args))))
            }
            "onepole" => {
                let input = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                let cutoff = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 20.0));
                Some(arena.alloc(Box::new(FunDSPNode::mono(
                    vec![input, cutoff],
                    Box::new(lowpole()),
                ))))
            }
            "lp" => {
                let input = self.parse_primary(arena)?;
                let cutoff = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 500.0));
                let resonance = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.707));
                Some(arena.alloc(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(lowpass()),
                ))))
            }
            "bp" => {
                let input = self.parse_primary(arena)?;
                let cutoff = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 500.0));
                let resonance = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.707));
                Some(arena.alloc(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(bandpass()),
                ))))
            }
            "hp" => {
                let input = self.parse_primary(arena)?;
                let cutoff = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 500.0));
                let resonance = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.707));
                Some(arena.alloc(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(highpass()),
                ))))
            }
            "moog" => {
                let input = self.parse_primary(arena)?;
                let cutoff = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 500.0));
                let resonance = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                Some(arena.alloc(Box::new(FunDSPNode::mono(
                    vec![input, cutoff, resonance],
                    Box::new(moog()),
                ))))
            }
            "delay" => {
                let input = self.parse_primary(arena)?;
                let delay_time = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 1000.0));
                Some(arena.alloc(Box::new(DelayNode::new(vec![input, delay_time]))))
            }
            "reverb" => {
                let input = self.parse_primary(arena)?;
                Some(arena.alloc(Box::new(FunDSPNode::stereo(
                    vec![input],
                    Box::new(reverb2_stereo(10.0, 2.0, 0.9, 1.0, lowpole_hz(18000.0))),
                ))))
            }
            "buf" => {
                let name = self.parse_str()?;
                let length = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, self.sample_rate as f32));
                let frames = vec![[0.0; 2]; length as usize];
                arena.store_buffer(name.clone(), frames);
                Some(arena.alloc(Box::new(BufRefNode::new(name))))
            }
            "file" => {
                let name = self.parse_str()?;
                let frames = get_audio_frames(&name);
                arena.store_buffer(name.clone(), frames);
                Some(arena.alloc(Box::new(BufRefNode::new(name))))
            }
            "play" => {
                let buf_ref_id = self.parse_primary(arena)?;
                let buf_name = arena.get(buf_ref_id).get_buf_name()?.to_string();
                let phase = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                Some(arena.alloc(Box::new(BufReaderNode::new(buf_name, vec![phase]))))
            }
            "tap" => {
                let buf_ref_id = self.parse_primary(arena)?;
                let buf_name = arena.get(buf_ref_id).get_buf_name()?.to_string();
                let offset = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                Some(arena.alloc(Box::new(BufTapNode::new(buf_name, vec![offset]))))
            }
            "rec" => {
                let buf_ref_id = self.parse_primary(arena)?;
                let buf_name = arena.get(buf_ref_id).get_buf_name()?.to_string();
                let input = self
                    .parse_primary(arena)
                    .unwrap_or_else(|| constant_node(arena, 0.0));
                Some(arena.alloc(Box::new(BufWriterNode::new(buf_name, vec![input]))))
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
