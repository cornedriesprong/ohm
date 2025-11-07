use crate::nodes::Node;
use crate::{graph::Graph, utils::get_audio_frames};
use fundsp::hacker32::*;
use petgraph::graph::NodeIndex;
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
    Pipe,
    Eof,
}

fn tokenize(str: String) -> VecDeque<Token> {
    let mut tokens = VecDeque::new();
    let mut chars = str.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\r' => {
                // ignore these
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
            '>' => {
                chars.next();
                if let Some(&'>') = chars.peek() {
                    chars.next();
                    tokens.push_back(Token::Pipe);
                } else {
                    tokens.push_back(Token::Greater);
                }
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
    env: HashMap<String, NodeIndex>,
    sample_rate: u32,
    graph: Graph,
}

impl Parser {
    pub(crate) fn new(src: String, sample_rate: u32) -> Self {
        return Self {
            tokens: tokenize(src),
            pos: 0,
            env: HashMap::new(),
            sample_rate,
            graph: Graph::new(),
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

    pub(crate) fn parse(mut self) -> Graph {
        let mut exprs = Vec::new();

        loop {
            while matches!(self.peek(), Token::Newline) {
                self.consume();
            }

            if matches!(self.peek(), Token::Eof) {
                break;
            }

            if let Some(expr) = self.parse_statement() {
                exprs.push(expr);
            }

            while matches!(self.peek(), Token::Newline) {
                self.consume();
            }
        }

        // if there is more than one top-level expression, mix them together at the output
        if exprs.len() > 1 {
            let mix_node = self.graph.add_node(Node::Mix);
            for &expr in &exprs {
                self.graph.connect_node(expr, mix_node);
            }
        }

        self.graph
    }

    fn parse_statement(&mut self) -> Option<NodeIndex> {
        if let Token::Identifier(name) = self.peek() {
            let name = name.clone();
            let prev_pos = self.pos;
            self.consume();

            if matches!(self.peek(), Token::Equal) {
                self.consume();
                if let Some(expr) = self.parse_expr(0) {
                    self.env.insert(name, expr);
                    return None;
                }
            }

            self.pos = prev_pos;
        }

        self.parse_expr(0)
    }

    fn parse_expr(&mut self, min_prec: u8) -> Option<NodeIndex> {
        let mut lhs = self.parse_primary()?;

        while let Some(op_prec) = self.op_precedence(self.peek()) {
            if op_prec < min_prec {
                break;
            }

            let op = self.consume();

            lhs = match op {
                Token::Pipe => {
                    if let Token::Identifier(name) = self.consume() {
                        self.parse_node(&name, Some(lhs))?
                    } else {
                        panic!("Expected function name after pipe operator");
                    }
                }
                _ => {
                    let rhs = self.parse_expr(op_prec + 1)?;
                    let node_idx = match op {
                        Token::Plus => self.graph.add_node(Node::Sum),
                        Token::Minus => self.graph.add_node(Node::Diff),
                        Token::Multiply => self.graph.add_node(Node::Gain),
                        Token::Divide => self.graph.add_node(Node::Divide),
                        Token::Modulo => self.graph.add_node(Node::Wrap),
                        Token::Power => self.graph.add_node(Node::Power),
                        Token::Greater => self.graph.add_node(Node::Greater),
                        Token::Less => self.graph.add_node(Node::Less),
                        Token::Equal => self.graph.add_node(Node::Equal),
                        _ => lhs,
                    };
                    if !matches!(
                        op,
                        Token::Plus
                            | Token::Minus
                            | Token::Multiply
                            | Token::Divide
                            | Token::Modulo
                            | Token::Power
                            | Token::Greater
                            | Token::Less
                            | Token::Equal
                    ) {
                        lhs
                    } else {
                        self.graph.connect_node(lhs, node_idx);
                        self.graph.connect_node(rhs, node_idx);
                        node_idx
                    }
                }
            };
        }

        Some(lhs)
    }

    fn parse_primary(&mut self) -> Option<NodeIndex> {
        let pos = self.pos;

        let result = match self.consume() {
            Token::Number(num) => Some(self.graph.add_node(Node::Constant(num))),
            Token::Minus => {
                if let Token::Number(num) = self.peek() {
                    let num = *num;
                    self.consume();
                    Some(self.graph.add_node(Node::Constant(-num)))
                } else {
                    None
                }
            }
            Token::Identifier(name) => {
                if let Some(op) = self.parse_node(&name, None) {
                    Some(op)
                } else if let Some(&node_id) = self.env.get(&name) {
                    Some(node_id)
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
        };

        // backtrack if we failed to parse
        if result.is_none() {
            self.pos = pos;
        }

        result
    }

    fn parse_str(&mut self) -> Option<String> {
        match self.consume() {
            Token::String(s) => Some(s),
            _ => None,
        }
    }

    fn parse_num(&mut self) -> Option<f32> {
        match self.consume() {
            Token::Number(n) => Some(n),
            _ => None,
        }
    }

    fn get_arg(&mut self, arg: Option<NodeIndex>, default: f32) -> NodeIndex {
        arg.or_else(|| self.parse_primary())
            .unwrap_or_else(|| self.graph.add_node(Node::Constant(default)))
    }

    fn create_node(&mut self, node: Node, args: Vec<NodeIndex>) -> NodeIndex {
        let node_idx = self.graph.add_node(node);
        for arg in args {
            self.graph.connect_node(arg, node_idx);
        }
        node_idx
    }

    fn create_fundsp_node(
        &mut self,
        audio_unit: Box<dyn AudioUnit>,
        is_stereo: bool,
        args: Vec<NodeIndex>,
    ) -> NodeIndex {
        let num_inputs = audio_unit.inputs();
        self.create_node(
            Node::FunDSP {
                audio_unit,
                is_stereo,
                input_buffer: vec![0.0; num_inputs],
            },
            args,
        )
    }

    fn parse_node(&mut self, name: &str, first_arg: Option<NodeIndex>) -> Option<NodeIndex> {
        match name {
            "ramp" => {
                let freq = self.get_arg(first_arg, 1.0);
                Some(self.create_node(
                    Node::Ramp {
                        phase: 0.0,
                        sample_rate: self.sample_rate,
                    },
                    vec![freq],
                ))
            }
            "sin" | "saw" | "sqr" | "tri" | "organ" | "softsaw" | "rossler" | "lorenz" => {
                let freq = self.get_arg(first_arg, 100.0);
                Some(self.create_fundsp_node(
                    match name {
                        "sin" => Box::new(sine()),
                        "saw" => Box::new(saw()),
                        "sqr" => Box::new(square()),
                        "tri" => Box::new(triangle()),
                        "organ" => Box::new(organ()),
                        "softsaw" => Box::new(soft_saw()),
                        "rossler" => Box::new(rossler()),
                        "lorenz" => Box::new(lorenz()),
                        _ => unreachable!(),
                    },
                    false,
                    vec![freq],
                ))
            }
            "noise" => Some(self.create_fundsp_node(Box::new(noise()), false, vec![])),
            "clip" => {
                let input = self.get_arg(first_arg, 0.0);
                Some(self.create_fundsp_node(Box::new(clip()), false, vec![input]))
            }
            "lfo" => {
                let freq = self.get_arg(first_arg, 1.0);
                Some(self.create_node(
                    Node::Lfo {
                        phase: 0.0,
                        sample_rate: self.sample_rate,
                    },
                    vec![freq],
                ))
            }
            "sh" => {
                let input = self.get_arg(first_arg, 0.0);
                let trig = self.get_arg(None, 0.0);
                Some(self.create_node(
                    Node::SampleAndHold {
                        value: [0.0; 2],
                        prev: 0.0,
                    },
                    vec![input, trig],
                ))
            }
            "pan" => {
                let input = self.get_arg(first_arg, 0.0);
                let pan = self.get_arg(None, 0.5);
                Some(self.create_fundsp_node(Box::new(panner()), true, vec![input, pan]))
            }
            "seq" => {
                let input = self.get_arg(first_arg, 0.0);
                let mut args = vec![input];
                while let Some(arg) = self.parse_primary() {
                    args.push(arg);
                }
                Some(self.create_node(Node::Seq, args))
            }
            "mix" => {
                let mut args = Vec::new();
                while let Some(arg) = self.parse_primary() {
                    args.push(arg);
                }
                Some(self.create_node(Node::Mix, args))
            }
            "onepole" => {
                let input = self.get_arg(first_arg, 0.0);
                let cutoff = self.get_arg(None, 1.0);
                Some(self.create_fundsp_node(Box::new(lowpole()), false, vec![input, cutoff]))
            }
            "lp" | "bp" | "hp" | "moog" => {
                let input = self.get_arg(first_arg, 0.0);
                let cutoff = self.get_arg(None, 500.0);
                let resonance = self.get_arg(None, 0.707);
                Some(self.create_fundsp_node(
                    match name {
                        "lp" => Box::new(lowpass()),
                        "bp" => Box::new(bandpass()),
                        "hp" => Box::new(highpass()),
                        "moog" => Box::new(moog()),
                        _ => unreachable!(),
                    },
                    false,
                    vec![input, cutoff, resonance],
                ))
            }
            "tanh" => {
                let input = self.get_arg(first_arg, 0.0);
                Some(self.create_fundsp_node(Box::new(shape(Tanh(1.0))), false, vec![input]))
            }
            "delay" => {
                let input = self.get_arg(first_arg, 0.0);
                let delay_time = self.get_arg(None, 100.0);
                Some(self.create_node(
                    Node::Delay {
                        buffer: Box::new([[0.0; 2]; 48000]),
                        write_pos: 0,
                    },
                    vec![input, delay_time],
                ))
            }
            "reverb" => {
                let input = self.get_arg(first_arg, 0.0);
                Some(self.create_fundsp_node(
                    Box::new(reverb2_stereo(10.0, 2.0, 0.9, 1.0, lowpole_hz(18000.0))),
                    true,
                    vec![input],
                ))
            }
            "buf" => {
                let length = self.parse_num().unwrap_or(self.sample_rate as f32) as usize;
                let frames = vec![[0.0; 2]; length];
                let node_idx = self.graph.add_buffer_node(frames);
                Some(node_idx)
            }
            "file" => {
                let filename = self.parse_str()?;
                let frames = get_audio_frames(&filename).ok()?;
                let node_idx = self.graph.add_buffer_node(frames);
                Some(node_idx)
            }
            "play" => {
                let phase = self.get_arg(first_arg, 0.0);
                let id = self.get_arg(None, 0.0);
                Some(self.create_node(Node::BufferReader { id }, vec![phase, id]))
            }
            "tap" => {
                let id = self.get_arg(None, 0.0);
                Some(self.create_node(Node::BufferTap { id, write_pos: 0 }, vec![]))
            }
            "rec" => {
                let id = self.get_arg(None, 0.0);
                Some(self.create_node(Node::BufferWriter { id, write_pos: 0 }, vec![]))
            }
            "log" => {
                let input = self.get_arg(first_arg, 0.0);
                Some(self.create_node(Node::Log, vec![input]))
            }
            _ => None,
        }
    }

    fn op_precedence(&self, token: &Token) -> Option<u8> {
        match token {
            Token::Pipe => Some(0),
            Token::Power => Some(1),
            Token::Multiply | Token::Divide | Token::Modulo => Some(2),
            Token::Plus | Token::Minus => Some(3),
            Token::Greater | Token::Less | Token::Equal => Some(4),
            _ => None,
        }
    }
}
