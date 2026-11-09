use std::{iter::Peekable, str::Chars};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Token {
    Num(f32),
    Ident(String),
    Ref(String),
    Plus,               // +
    Minus,              // -
    Asterisk,           // *
    Slash,              // /
    Assignment,         // =
    Equal,              // ==
    NotEqual,           // !=
    Bar,                // |
    Arrow,              // ->
    LParen,             // (
    RParen,             // )
    LBracket,           // [
    RBracket,           // ]
    LCurly,             // {
    RCurly,             // }
    GreaterThan,        // >
    LessThan,           // <
    GreaterThanOrEqual, // >=
    LessThanOrEqual,    // <=
    Caret,              // ^
    Percent,            // %
    Newline,
    // leading whitespace on a line other than the first
    Indent,
    Eof,
}

#[derive(Debug, PartialEq)]
pub(crate) enum LexError {
    InvalidNumber(String),
    UnrecognizedCharacter(char),
}

pub(crate) struct Lexer<'a> {
    src: Peekable<Chars<'a>>,
    line_start: bool,
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(src: &'a str) -> Self {
        Self {
            src: src.chars().peekable(),
            line_start: false,
        }
    }

    pub(crate) fn next(&mut self) -> Result<Token, LexError> {
        let indented = self.line_start && matches!(self.src.peek(), Some(' ' | '\t'));
        self.line_start = false;

        while let Some(' ' | '\r' | '\t' | ',') = self.src.peek() {
            self.src.next();
        }

        // skip comments
        let mut ahead = self.src.clone();
        if let (Some('/'), Some('/')) = (ahead.next(), ahead.next()) {
            while let Some(&ch) = self.src.peek() {
                if ch == '\n' {
                    break;
                }
                self.src.next();
            }
        }

        // blank lines aren't indented
        if indented && !matches!(self.src.peek(), None | Some('\n')) {
            return Ok(Token::Indent);
        }

        if let Some(ch) = self.src.next() {
            match ch {
                '0'..='9' | '.' => self.parse_num(ch),
                'a'..='z' | 'A'..='Z' | '_' => Ok(self.parse_ident(ch)),
                '+' => Ok(Token::Plus),
                '-' => {
                    if let Some('>') = self.src.peek() {
                        self.src.next();
                        Ok(Token::Arrow)
                    } else {
                        Ok(Token::Minus)
                    }
                }
                '*' => Ok(Token::Asterisk),
                '/' => Ok(Token::Slash),
                '|' => Ok(Token::Bar),
                '(' => Ok(Token::LParen),
                ')' => Ok(Token::RParen),
                '[' => Ok(Token::LBracket),
                ']' => Ok(Token::RBracket),
                '{' => Ok(Token::LCurly),
                '}' => Ok(Token::RCurly),
                '<' => {
                    if let Some('=') = self.src.peek() {
                        self.src.next();
                        Ok(Token::LessThanOrEqual)
                    } else {
                        Ok(Token::LessThan)
                    }
                }
                '^' => Ok(Token::Caret),
                '%' => Ok(Token::Percent),
                '=' => {
                    if let Some('=') = self.src.peek() {
                        self.src.next();
                        Ok(Token::Equal)
                    } else {
                        Ok(Token::Assignment)
                    }
                }
                '!' => {
                    if let Some('=') = self.src.peek() {
                        self.src.next();
                        Ok(Token::NotEqual)
                    } else {
                        Err(LexError::UnrecognizedCharacter(ch))
                    }
                }
                '>' => {
                    if let Some('=') = self.src.peek() {
                        self.src.next();
                        Ok(Token::GreaterThanOrEqual)
                    } else {
                        Ok(Token::GreaterThan)
                    }
                }
                '~' => {
                    let mut ident = String::new();
                    while let Some(&ch) = self.src.peek() {
                        if ch.is_alphanumeric() || ch == '_' {
                            ident.push(ch);
                            self.src.next();
                        } else {
                            break;
                        }
                    }
                    Ok(Token::Ref(ident))
                }
                '\n' => {
                    self.line_start = true;
                    Ok(Token::Newline)
                }
                _ => Err(LexError::UnrecognizedCharacter(ch)),
            }
        } else {
            Ok(Token::Eof)
        }
    }

    fn parse_num(&mut self, ch: char) -> Result<Token, LexError> {
        let mut num_str = ch.to_string();
        while let Some(&ch) = self.src.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                num_str.push(ch);
                self.src.next();
            } else {
                break;
            }
        }
        num_str
            .parse::<f32>()
            .map(Token::Num)
            .map_err(|_| LexError::InvalidNumber(num_str))
    }

    fn parse_ident(&mut self, ch: char) -> Token {
        let mut ident = ch.to_string();
        while let Some(&ch) = self.src.peek() {
            // `#` for sharps, e.g. `c#4`
            if ch.is_alphanumeric() || ch == '_' || ch == '#' {
                ident.push(ch);
                self.src.next();
            } else {
                break;
            }
        }
        Token::Ident(ident)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Token::*;

    fn lex_src(src: &str) -> Result<Token, LexError> {
        let mut lexer = Lexer::new(src);
        lexer.next()
    }

    #[test]
    fn lex_number() -> Result<(), LexError> {
        assert_eq!(lex_src("0")?, Num(0.0));
        assert_eq!(lex_src("10")?, Num(10.0));
        assert_eq!(lex_src("001")?, Num(1.0));
        assert_eq!(lex_src("1.5")?, Num(1.5));
        assert_eq!(lex_src("0.5")?, Num(0.5));
        assert_eq!(lex_src("0.25")?, Num(0.25));
        assert_eq!(lex_src(".5")?, Num(0.5));
        assert_eq!(lex_src("1.")?, Num(1.0));
        assert_eq!(lex_src("123")?, Num(123.0));
        assert_eq!(lex_src("1a")?, Num(1.0));
        assert_eq!(lex_src("10a")?, Num(10.0));

        Ok(())
    }

    fn assert_invalid_number(src: &str) {
        match lex_src(src) {
            Err(LexError::InvalidNumber(str)) => assert_eq!(str, src),
            other => panic!("Expected invalid number, got {other:?}"),
        }
    }

    #[test]
    fn lex_invalid_number() {
        assert_invalid_number("1.2.3");
        assert_invalid_number(".");
    }

    #[test]
    fn lex_unrecognized_character() {
        for ch in ['$', '!'] {
            match lex_src(&ch.to_string()) {
                Err(LexError::UnrecognizedCharacter(got)) => assert_eq!(got, ch),
                other => panic!("Expected unrecognized character, got {other:?}"),
            }
        }
    }

    #[test]
    fn lex_identifier() -> Result<(), LexError> {
        assert_eq!(lex_src("yo")?, Ident("yo".to_string()));
        assert_eq!(lex_src("yo ")?, Ident("yo".to_string()));
        assert_eq!(lex_src(" yo")?, Ident("yo".to_string()));
        assert_eq!(lex_src(" yo ")?, Ident("yo".to_string()));
        assert_eq!(lex_src("Yo")?, Ident("Yo".to_string()));
        assert_eq!(lex_src("Yo_2")?, Ident("Yo_2".to_string()));
        assert_eq!(lex_src("_yo")?, Ident("_yo".to_string()));
        assert_eq!(lex_src("c#4")?, Ident("c#4".to_string()));

        Ok(())
    }

    #[test]
    fn lex_ref() -> Result<(), LexError> {
        assert_eq!(lex_src("~x")?, Ref("x".to_string()));

        Ok(())
    }

    #[test]
    fn lex_symbols() -> Result<(), LexError> {
        assert_eq!(lex_src("+")?, Plus);
        assert_eq!(lex_src("-")?, Minus);
        assert_eq!(lex_src("*")?, Asterisk);
        assert_eq!(lex_src("/")?, Slash);
        assert_eq!(lex_src("=")?, Assignment);
        assert_eq!(lex_src("==")?, Equal);
        assert_eq!(lex_src("!=")?, NotEqual);
        assert_eq!(lex_src("|")?, Bar);
        assert_eq!(lex_src("->")?, Arrow);
        assert_eq!(lex_src(">")?, GreaterThan);
        assert_eq!(lex_src("<")?, LessThan);
        assert_eq!(lex_src(">=")?, GreaterThanOrEqual);
        assert_eq!(lex_src("<=")?, LessThanOrEqual);
        assert_eq!(lex_src("(")?, LParen);
        assert_eq!(lex_src(")")?, RParen);
        assert_eq!(lex_src("[")?, LBracket);
        assert_eq!(lex_src("]")?, RBracket);
        assert_eq!(lex_src("{")?, LCurly);
        assert_eq!(lex_src("}")?, RCurly);
        assert_eq!(lex_src("^")?, Caret);
        assert_eq!(lex_src("%")?, Percent);
        assert_eq!(lex_src("\n")?, Newline);
        assert_eq!(lex_src("")?, Eof);

        Ok(())
    }

    #[test]
    fn lex_indent() -> Result<(), LexError> {
        let lex_all = |src| {
            let mut lexer = Lexer::new(src);
            let mut tokens = Vec::new();
            loop {
                match lexer.next()? {
                    Eof => return Ok::<_, LexError>(tokens),
                    token => tokens.push(token),
                }
            }
        };
        assert_eq!(
            lex_all("a\n  b")?,
            vec![Ident("a".into()), Newline, Indent, Ident("b".into())]
        );

        // blank and comment lines aren't indented
        assert_eq!(
            lex_all("a\n  \n  // hey\nb")?,
            vec![
                Ident("a".into()),
                Newline,
                Newline,
                Newline,
                Ident("b".into())
            ]
        );

        Ok(())
    }

    #[test]
    fn lex_comment() -> Result<(), LexError> {
        assert_eq!(lex_src("// hey")?, Token::Eof);
        assert_eq!(lex_src("// TODO: fix")?, Token::Eof);

        // a single slash is a division
        assert_eq!(lex_src("/ 2")?, Token::Slash);

        Ok(())
    }
}
