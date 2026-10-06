/// Exact-match helper for `tensor_map.rs` files.
///
/// Every model crate maps checkpoint names to GGUF names with a first
/// exact-match block plus custom prefix/layer logic. This helper covers the
/// exact part so each file keeps one `TABLE` instead of a 15-arm `match`.
///
/// Returns `Some(mapped)` on hit, `None` on miss (caller falls through to
/// prefix logic or returns `None` for skips).
pub fn map_with_table(name: &str, table: &[(&str, &str)]) -> Option<String> {
    table
        .iter()
        .find(|(from, _)| *from == name)
        .map(|(_, to)| (*to).to_string())
}

/// Table-driven exact match with a fallback closure for prefix/layer rules.
///
/// Example (TimesFM head):
/// ```rust
/// const HEAD: &[(&str, &str)] = &[("a.weight", "b.weight")];
/// let out = zsfm_gguf::map_with_table("a.weight", HEAD);
/// assert_eq!(out.as_deref(), Some("b.weight"));
/// ```
#[macro_export]
macro_rules! tensor_map {
    ($name:expr, { $($from:literal => $to:literal),* $(,)? } $(, $fallback:expr)?) => {{
        const TABLE: &[(&str, &str)] = &[$(($from, $to)),*];
        match $crate::map_with_table($name, TABLE) {
            Some(mapped) => Some(mapped),
            None => {
                $( $fallback )?
                None::<String>
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_hits() {
        const T: &[(&str, &str)] = &[("a.weight", "b.weight")];
        assert_eq!(map_with_table("a.weight", T).as_deref(), Some("b.weight"));
        assert_eq!(map_with_table("missing", T), None);
    }

    #[test]
    fn macro_form() {
        let out: Option<String> = tensor_map!("a.weight", {"a.weight" => "b.weight"});
        assert_eq!(out.as_deref(), Some("b.weight"));
    }
}
