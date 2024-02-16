use regex::Regex;
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn clx(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

struct Token {
    text: String,
}

fn tokenize(text: &str) -> Vec<Token> {
    Regex::new(
        concat!(
            "[\\s,]*",
            "(",
                "~@", "|",
                "[\\[\\]{}()'`~^@]", "|",
                "\"(?:[\\\\].", "|", "[^\\\\\"])*\"?", "|",
                ";.*", "|",
                "[^\\s\\[\\]{}()'\"`@,;]+",
            ")"))
        .unwrap()
        .captures_iter(text)
        .map(|cap| Token{ text: cap[1].to_string() })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_string() {
        let _tokenize = |text: &str| tokenize(text)
            .iter()
            .map(|t| t.text.clone())
            .collect::<Vec<_>>();
        assert_eq!(_tokenize("    "), Vec::<String>::new());
        assert_eq!(_tokenize(" ~@ "), vec!["~@"]);
        assert_eq!(_tokenize("  \"hello\n world!\"   "),
            vec!["\"hello\n world!\""]);
        assert_eq!(_tokenize("(foo :bar [1 \"hello\"] {:baz 'quux})"),
            vec!["(", "foo", ":bar",
                "[", "1", "\"hello\"", "]",
                "{", ":baz", "'", "quux", "}", ")"]);
    }
}
