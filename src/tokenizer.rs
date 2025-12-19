use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub(crate) enum Token {
    Number(f32),
    Identifier(String),
    String(String),
    Assign,
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
    Chain,
    Pipe,
    Tilde,
    Eof,
}

pub(crate) fn tokenize(str: String) -> VecDeque<Token> {
    let mut tokens = VecDeque::new();
    let mut chars = str.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\r' => {
                // ignore these
                chars.next();
            }
            '#' => {
                while let Some(&ch) = chars.peek() {
                    chars.next();
                    if ch == '\n' {
                        tokens.push_back(Token::Newline);
                        break;
                    }
                }
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
                chars.next();
                if let Some(&'=') = chars.peek() {
                    chars.next();
                    tokens.push_back(Token::Equal);
                } else {
                    tokens.push_back(Token::Assign);
                }
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
                    tokens.push_back(Token::Chain);
                } else {
                    tokens.push_back(Token::Greater);
                }
            }
            '|' => {
                tokens.push_back(Token::Pipe);
                chars.next();
            }
            '~' => {
                tokens.push_back(Token::Tilde);
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
