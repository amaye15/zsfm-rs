//! `CategoricalOrdinalEncoder` (`cat_encoder_mode="appearance"`, the wrapper's default): assigns
//! integer codes to a categorical column's values in order of first appearance in the training
//! data; unknown/missing values (at fit or transform time) map to `-1`.

use std::collections::HashMap;

use serde_json::Value;

pub struct CategoricalOrdinalEncoder {
    index: HashMap<String, i64>,
}

impl CategoricalOrdinalEncoder {
    pub fn fit(column: &[Value]) -> Self {
        let mut index = HashMap::new();
        let mut next = 0i64;
        for v in column {
            if is_missing(v) {
                continue;
            }
            let key = value_key(v);
            index.entry(key).or_insert_with(|| {
                let code = next;
                next += 1;
                code
            });
        }
        CategoricalOrdinalEncoder { index }
    }

    pub fn transform(&self, column: &[Value]) -> Vec<f64> {
        column
            .iter()
            .map(|v| {
                if is_missing(v) {
                    return -1.0;
                }
                *self.index.get(&value_key(v)).unwrap_or(&-1) as f64
            })
            .collect()
    }
}

/// Classification label encoder (`CategoricalOrdinalEncoder(mode="alphabetical")` as used on
/// `y`): unique labels sorted alphabetically (by their string form) get codes `0..n_classes-1`.
pub struct LabelEncoder {
    classes: Vec<String>, // sorted; index = class code
}

impl LabelEncoder {
    pub fn fit(y: &[Value]) -> Self {
        let mut uniq: Vec<String> = y.iter().filter(|v| !is_missing(v)).map(value_key).collect();
        uniq.sort();
        uniq.dedup();
        LabelEncoder { classes: uniq }
    }

    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    pub fn transform(&self, y: &[Value]) -> Vec<f64> {
        y.iter()
            .map(|v| {
                if is_missing(v) {
                    return -1.0;
                }
                let key = value_key(v);
                self.classes.iter().position(|c| c == &key).map(|i| i as f64).unwrap_or(-1.0)
            })
            .collect()
    }

    pub fn decode(&self, code: usize) -> &str {
        &self.classes[code]
    }
}

fn is_missing(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.is_empty() || s.eq_ignore_ascii_case("nan"),
        Value::Number(n) => n.as_f64().map(f64::is_nan).unwrap_or(false),
        _ => false,
    }
}

fn value_key(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_appearance_order() {
        let col = vec![
            Value::String("blue".into()),
            Value::String("red".into()),
            Value::String("blue".into()),
            Value::String("green".into()),
        ];
        let enc = CategoricalOrdinalEncoder::fit(&col);
        assert_eq!(enc.transform(&col), vec![0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_unknown_at_transform_time() {
        let train = vec![Value::String("a".into()), Value::String("b".into())];
        let enc = CategoricalOrdinalEncoder::fit(&train);
        let test = vec![Value::String("a".into()), Value::String("c".into())];
        assert_eq!(enc.transform(&test), vec![0.0, -1.0]);
    }

    #[test]
    fn test_missing_values() {
        let col = vec![Value::String("a".into()), Value::Null, Value::String("a".into())];
        let enc = CategoricalOrdinalEncoder::fit(&col);
        assert_eq!(enc.transform(&col), vec![0.0, -1.0, 0.0]);
    }
}
