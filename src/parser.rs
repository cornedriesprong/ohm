use crate::container::Graph;
use crate::nodes::Node;
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
        if exprs.len() == 1 {
            // Root is already in the graph
        } else if !exprs.is_empty() {
            println!("Mixing {} top-level expressions", exprs.len());
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
        match self.consume() {
            Token::Number(num) => Some(self.graph.add_node(Node::Constant(num))),
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
        }
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

    fn parse_node(&mut self, name: &str, first_arg: Option<NodeIndex>) -> Option<NodeIndex> {
        match name {
            "ramp" => {
                let freq = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(1.0)));
                let node_idx = self.graph.add_node(Node::Ramp {
                    phase: 0.0,
                    sample_rate: self.sample_rate,
                });
                self.graph.connect_node(freq, node_idx);
                Some(node_idx)
            }
            "sin" => {
                let freq = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(100.0)));
                let node = Box::new(sine());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(freq, node_idx);
                Some(node_idx)
            }
            "saw" => {
                let freq = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(100.0)));
                let node = Box::new(saw());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(freq, node_idx);
                Some(node_idx)
            }
            "sqr" => {
                let freq = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(100.0)));
                let node = Box::new(square());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(freq, node_idx);
                Some(node_idx)
            }
            "tri" => {
                let freq = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(100.0)));
                let node = Box::new(triangle());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(freq, node_idx);
                Some(node_idx)
            }
            "noise" => {
                let node = Box::new(noise());
                let num_inputs = node.inputs();
                Some(self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                }))
            }
            "clip" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let node = Box::new(clip());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(input, node_idx);
                Some(node_idx)
            }
            "lfo" => {
                let freq = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(1.0)));
                let node_idx = self.graph.add_node(Node::Lfo {
                    phase: 0.0,
                    sample_rate: self.sample_rate,
                });
                self.graph.connect_node(freq, node_idx);
                Some(node_idx)
            }
            "sh" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let trig = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let node_idx = self.graph.add_node(Node::SampleAndHold {
                    value: [0.0; 2],
                    prev: 0.0,
                });
                self.graph.connect_node(input, node_idx);
                self.graph.connect_node(trig, node_idx);
                Some(node_idx)
            }
            "pan" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let pan = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.5)));
                let node = Box::new(panner());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(input, node_idx);
                self.graph.connect_node(pan, node_idx);
                Some(node_idx)
            }
            "seq" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let mut args = vec![input];
                while let Some(arg) = self.parse_primary() {
                    args.push(arg);
                }
                let node_idx = self.graph.add_node(Node::Seq);
                for arg in args {
                    self.graph.connect_node(arg, node_idx);
                }
                Some(node_idx)
            }
            "mix" => {
                let mut args = Vec::new();
                while let Some(arg) = self.parse_primary() {
                    args.push(arg);
                }
                let node_idx = self.graph.add_node(Node::Mix);
                for arg in args {
                    self.graph.connect_node(arg, node_idx);
                }
                Some(node_idx)
            }
            "onepole" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let cutoff = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(20.0)));
                let node = Box::new(lowpole());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(input, node_idx);
                self.graph.connect_node(cutoff, node_idx);
                Some(node_idx)
            }
            "lp" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let cutoff = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(500.0)));
                let resonance = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.707)));
                let node = Box::new(lowpass());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(input, node_idx);
                self.graph.connect_node(cutoff, node_idx);
                self.graph.connect_node(resonance, node_idx);
                Some(node_idx)
            }
            "bp" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let cutoff = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(500.0)));
                let resonance = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.707)));
                let node = Box::new(bandpass());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(input, node_idx);
                self.graph.connect_node(cutoff, node_idx);
                self.graph.connect_node(resonance, node_idx);
                Some(node_idx)
            }
            "hp" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let cutoff = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(500.0)));
                let resonance = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.707)));
                let node = Box::new(highpass());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(input, node_idx);
                self.graph.connect_node(cutoff, node_idx);
                self.graph.connect_node(resonance, node_idx);
                Some(node_idx)
            }
            "moog" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let cutoff = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(500.0)));
                let resonance = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let node = Box::new(moog());
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: false,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(input, node_idx);
                self.graph.connect_node(cutoff, node_idx);
                self.graph.connect_node(resonance, node_idx);
                Some(node_idx)
            }
            "delay" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let delay_time = self
                    .parse_primary()
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(1000.0)));
                let node_idx = self.graph.add_node(Node::Delay {
                    buffer: Box::new([[0.0; 2]; 48000]),
                    write_pos: 0,
                });
                self.graph.connect_node(input, node_idx);
                self.graph.connect_node(delay_time, node_idx);
                Some(node_idx)
            }
            "reverb" => {
                let input = first_arg
                    .or_else(|| self.parse_primary())
                    .unwrap_or_else(|| self.graph.add_node(Node::Constant(0.0)));
                let node = Box::new(reverb2_stereo(10.0, 2.0, 0.9, 1.0, lowpole_hz(18000.0)));
                let num_inputs = node.inputs();
                let node_idx = self.graph.add_node(Node::FunDSP {
                    node,
                    is_stereo: true,
                    input_buffer: vec![0.0; num_inputs],
                });
                self.graph.connect_node(input, node_idx);
                Some(node_idx)
            }
            "buf" => {
                // TODO: implement buffer support
                None
            }
            "file" => {
                // TODO: implement buffer support
                None
            }
            "play" => {
                // TODO: implement buffer support
                None
            }
            "tap" => {
                // TODO: implement buffer support
                None
            }
            "rec" => {
                // TODO: implement buffer support
                None
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
