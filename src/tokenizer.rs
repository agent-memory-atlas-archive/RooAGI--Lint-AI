//! Shared query/document tokenization for the primary lexical/rerank path
//! (`crate::index`) and the experimental segment-routing path
//! (`crate::segments`).
//!
//! The two callers intentionally use different rules. This is not
//! duplication to collapse into one behavior. A LongMemEval-S benchmark
//! (500 scoped queries for the primary path, 133 multi-session queries for
//! segment routing) showed each mode is a real, measured improvement for
//! its own caller and a regression for the other:
//!
//! - Switching segment routing from `Stemmed` to `Unstemmed` cost ~5pp of
//!   recall@5 and ~2pp of MRR on segment-routing variants.
//! - Switching the primary path from `Unstemmed` to `Stemmed` cost
//!   ~0.1-0.3pp across recall/MRR/NDCG.
//!
//! Keep both modes and pick per caller; don't unify them.

use crate::lang::Lang;
use crate::query_expansion::normalize_for_index;
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerMode {
    /// Regex-bounded terms (`[A-Za-z][A-Za-z0-9_-]{2,}`, min length 3),
    /// lowercased, not stemmed. Used by `crate::index`'s lexical/rerank
    /// path.
    Unstemmed,
    /// Terms split on non-alphanumeric boundaries (min length 2), each
    /// stemmed with an English Porter stemmer via
    /// [`normalize_for_index`]. Used by `crate::segments`'s routing path.
    Stemmed,
}

/// Latin letter class for token regexes: ASCII plus accented Latin
/// (Latin-1 Supplement U+00C0–U+00FF and Latin Extended-A U+0100–U+017F),
/// so Spanish/French/etc. words tokenize as units ("niño" is one token,
/// not "ni"). Pure-ASCII input matches byte-identically to `[A-Za-z]`.
pub(crate) const LATIN_LETTER: &str = r"A-Za-zÀ-ÿĀ-ſ";

/// Tokenizes `input` according to `mode`. Order matches input order and
/// duplicates are preserved; callers that need a set should collect into
/// one (as `crate::segments::query_tokens` does).
pub fn tokenize(input: &str, mode: TokenizerMode) -> Vec<String> {
    match mode {
        TokenizerMode::Unstemmed => unstemmed_tokens(input),
        TokenizerMode::Stemmed => stemmed_tokens(input),
    }
}

/// True if `token` is a stopword under `mode`. `token` must already be
/// tokenized/normalized under the same mode. The two stopword lists use
/// different vocabularies (raw words vs. stemmed fragments) and are not
/// interchangeable.
pub fn is_stopword(token: &str, mode: TokenizerMode) -> bool {
    match mode {
        TokenizerMode::Unstemmed => unstemmed_stopwords().contains(token),
        TokenizerMode::Stemmed => stemmed_stopwords().contains(token),
    }
}

fn unstemmed_tokens(input: &str) -> Vec<String> {
    static TOKEN_RE: OnceLock<Regex> = OnceLock::new();
    let token_re = TOKEN_RE.get_or_init(|| {
        Regex::new(&format!(r"[{L}][{L}0-9_\-]{{2,}}", L = LATIN_LETTER)).expect("valid regex")
    });
    token_re
        .find_iter(input)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn stemmed_tokens(input: &str) -> Vec<String> {
    input
        .split(|ch: char| !ch.is_alphanumeric())
        .map(normalize_for_index)
        .filter(|token| token.len() > 1)
        .collect()
}

fn unstemmed_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        [
            "how",
            "many",
            "much",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "did",
            "does",
            "have",
            "has",
            "had",
            "been",
            "being",
            "was",
            "were",
            "are",
            "the",
            "and",
            "or",
            "for",
            "from",
            "with",
            "that",
            "this",
            "these",
            "those",
            "currently",
            "recently",
            "past",
            "last",
            "next",
            "into",
            "onto",
            "about",
            "after",
            "before",
            "over",
            "under",
            "between",
            "during",
            "i",
            "you",
            "we",
            "they",
        ]
        .into_iter()
        .collect()
    })
}

fn stemmed_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        [
            "a", "an", "and", "are", "can", "did", "do", "doe", "for", "from", "had", "have",
            "how", "i", "in", "is", "it", "many", "mani", "me", "my", "of", "on", "or", "that",
            "the", "thi", "this", "to", "wa", "what", "when", "where", "which", "who", "with",
            "you",
        ]
        .into_iter()
        .collect()
    })
}

/// Spanish stopwords. Listed in BOTH surface forms they are checked
/// against, because callers normalize differently:
/// - raw-lowercased with accents ("dónde", "está") — `literal_query_tokens`
///   tokenizes the raw query;
/// - deunicoded + English-Porter-stemmed ("donde", "esta") — query-term
///   paths that run through `normalize_for_index` first
///   (`routing_content_terms`, `identify_focus`, `expand_query_terms`).
/// Membership is a set lookup, so carrying both forms is harmless.
/// Only consulted when the caller passes `Lang::Es` — English text never
/// sees these (surface forms like "no"/"son"/"era" would otherwise
/// collide with English words).
fn spanish_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        [
            // Raw (accented) forms.
            "el", "la", "los", "las", "lo", "un", "una", "unos", "unas", "del", "al", "de", "en",
            "a", "por", "para", "con", "sin", "sobre", "entre", "hasta", "desde", "hacia",
            "durante", "mediante", "según", "tras", "ante", "bajo", "contra", "y", "e", "ni", "o",
            "u", "pero", "porque", "aunque", "sino", "pues", "que", "yo", "tú", "él", "ella",
            "ello", "nosotros", "nosotras", "vosotros", "vosotras", "ellos", "ellas", "me", "te",
            "se", "nos", "les", "mi", "mis", "tus", "su", "sus", "nuestro", "nuestra", "vuestra",
            "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella",
            "aquellos", "aquellas", "esto", "eso", "aquello", "no", "sí", "también", "tampoco",
            "muy", "tan", "más", "menos", "mucho", "mucha", "muchos", "muchas", "poco", "poca",
            "bastante", "además", "así", "aquí", "ahí", "allí", "ahora", "hoy", "ayer", "mañana",
            "siempre", "nunca", "jamás", "ya", "todavía", "aún", "entonces", "luego", "después",
            "antes", "es", "son", "era", "eran", "fue", "fueron", "sea", "sean", "está", "están",
            "estaba", "estaban", "estoy", "hay", "ha", "han", "he", "hemos", "tiene", "tienen",
            "hace", "hacen", "puede", "pueden", "debe", "deben", "qué", "quién", "quiénes", "cuál",
            "cuáles", "dónde", "cuándo", "cuánto", "cuánta", "cuántos", "cuántas", "cómo", "donde",
            "cuando", "cual", "cuales", "quien", "quienes", "como", "cuanto", "cuanta", "cuantos",
            "cuantas",
            // Deunicoded + English-stemmed forms: exactly what
            // `normalize_for_index` produces for the raw words above
            // (verified by `spanish_stopwords_cover_normalized_forms`).
            // Note the Porter stemmer is aggressive on Spanish
            // ("dónde" -> "dond", "este" -> "est"); these are the empirical
            // outputs, not hand derivations.
            "adema", "ahi", "ahora", "al", "alli", "ant", "aquel", "aquella", "aquello", "aqui",
            "asi", "aun", "aunqu", "ayer", "bajo", "bastant", "como", "con", "contra", "cual",
            "cuando", "cuanta", "cuanto", "de", "debe", "deben", "del", "desd", "despu", "dond",
            "durant", "el", "ella", "ello", "en", "entonc", "entr", "era", "eran", "es", "esa",
            "ese", "eso", "est", "esta", "estaba", "estaban", "estan", "esto", "estoy", "fue",
            "fueron", "ha", "hace", "hacen", "hacia", "han", "hasta", "hay", "he", "hemo", "hoy",
            "jama", "la", "las", "le", "les", "lo", "los", "luego", "manana", "mas", "me",
            "mediant", "meno", "mi", "mis", "mucha", "mucho", "muy", "ni", "no", "nos", "nosotra",
            "nosotro", "nuestra", "nuestro", "nunca", "para", "pero", "poca", "poco", "por",
            "porqu", "pu", "pue", "pued", "pueden", "que", "quien", "se", "sea", "sean", "segun",
            "si", "siempr", "sin", "sino", "sobr", "son", "su", "sus", "tambien", "tampoco", "tan",
            "te", "tien", "tienen", "todavia", "tra", "tras", "tu", "tus", "un", "una", "uno",
            "vosotra", "vosotro", "vuestra", "ya", "yo",
        ]
        .into_iter()
        .collect()
    })
}

/// True if `token` is a stopword for `lang` under `mode`. English behavior
/// is unchanged (`is_stopword`); Spanish adds its function words on top.
/// `lang` must already be resolved — `Auto` falls back to English.
pub fn is_stopword_for_lang(token: &str, mode: TokenizerMode, lang: Lang) -> bool {
    if is_stopword(token, mode) {
        return true;
    }
    match lang {
        Lang::Es => spanish_stopwords().contains(token),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unstemmed_matches_manual_regex() {
        let re = Regex::new(&format!(r"[{L}][{L}0-9_\-]{{2,}}", L = LATIN_LETTER)).unwrap();
        let samples = [
            "What degree did I graduate with?",
            "How many miles did I run last week?",
            "I cooked pasta and watched a movie.",
        ];
        for s in samples {
            let expected: Vec<String> =
                re.find_iter(s).map(|m| m.as_str().to_lowercase()).collect();
            let actual = tokenize(s, TokenizerMode::Unstemmed);
            assert_eq!(expected, actual, "mismatch for {s:?}");
        }
    }

    #[test]
    fn unstemmed_keeps_spanish_accents() {
        // Accented words must tokenize as units (previously "niño" yielded
        // zero tokens and "está" was truncated to "est").
        assert_eq!(
            tokenize("¿Dónde está la biblioteca?", TokenizerMode::Unstemmed),
            vec!["dónde", "está", "biblioteca"]
        );
        assert_eq!(
            tokenize("El niño juega", TokenizerMode::Unstemmed),
            vec!["niño", "juega"]
        );
    }

    #[test]
    fn spanish_stopwords_cover_normalized_forms() {
        // Fixed-point check: every Spanish stopword, once run through the
        // index-time normalization, must land back in the set (or vanish).
        // Otherwise the normalized query-term paths would miss it.
        for word in spanish_stopwords().iter() {
            let normalized = normalize_for_index(word);
            for form in normalized.split_whitespace() {
                assert!(
                    spanish_stopwords().contains(form),
                    "normalized form {form:?} of {word:?} missing from Spanish stopwords"
                );
            }
        }
    }

    #[test]
    fn spanish_stopwords_do_not_leak_into_english() {
        // The per-language gate: English text never consults the Spanish
        // list, so colliding surface forms ("no", "son", "era") stay live
        // in English.
        for word in ["no", "son", "era", "tan", "la", "el"] {
            assert!(
                !is_stopword(word, TokenizerMode::Unstemmed),
                "{word} must not be an English stopword"
            );
            assert!(
                is_stopword_for_lang(word, TokenizerMode::Unstemmed, Lang::Es),
                "{word} should be a Spanish stopword"
            );
            assert!(
                !is_stopword_for_lang(word, TokenizerMode::Unstemmed, Lang::En),
                "{word} must not be filtered for English"
            );
        }
    }

    #[test]
    fn unstemmed_stopword_matches_original_list() {
        for word in ["how", "many", "does", "was", "the", "and"] {
            assert!(
                is_stopword(word, TokenizerMode::Unstemmed),
                "{word} should be a stopword"
            );
        }
        for word in ["degree", "graduate", "miles", "pasta"] {
            assert!(
                !is_stopword(word, TokenizerMode::Unstemmed),
                "{word} should not be a stopword"
            );
        }
    }

    #[test]
    fn stemmed_stopword_matches_segment_router_list() {
        for word in ["doe", "mani", "thi", "wa", "what"] {
            assert!(
                is_stopword(word, TokenizerMode::Stemmed),
                "{word} should be a stopword"
            );
        }
        for word in ["degre", "graduat", "mile", "pasta"] {
            assert!(
                !is_stopword(word, TokenizerMode::Stemmed),
                "{word} should not be a stopword"
            );
        }
    }
}
