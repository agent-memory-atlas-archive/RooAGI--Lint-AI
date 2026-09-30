//! Latin-script tokenizer for tantivy's TEXT fields.
//!
//! Registered as an override of the built-in `"default"` tokenizer on every
//! index lint-ai creates (see [`register_latin_tokenizer`]). Overriding the
//! name rather than naming a per-field tokenizer keeps the schema unchanged
//! and applies to already-existing on-disk indexes too: the tokenizer name
//! is resolved through the index's `TokenizerManager` at index and query
//! time.
//!
//! Behavior: replicates tantivy's default tokenizer exactly (split on
//! non-alphanumeric, drop tokens >= 40 bytes, lowercase) plus one step —
//! [`deunicode`] transliteration of each token ("niño" -> "nino").
//!
//! Why: `normalize_for_index` deunicodes Latin text, so the boosted fields
//! (`entities`, `important_terms`) hold ASCII-folded tokens. If the
//! content/headings fields kept raw accented tokens, a query for "niño"
//! would match `content` ("niño") but never the boosted fields ("nino"),
//! silently dropping the 2.0–2.4x entity/term boosts for every accented
//! term. Folding at the tokenizer level keeps index and query in agreement
//! on every TEXT field, for both the index side and the QueryParser side
//! (which resolves the field tokenizer through the same manager).
//! Pure-ASCII text indexes byte-identically to before this change
//! (deunicode is the identity on ASCII).
//!
//! MERGE NOTE (parallel zh branch): the zh branch registers its own
//! `"default"` override (`CjkTokenizer`: Han runs -> character bigrams,
//! other runs -> default-tokenizer replication). The two overrides target
//! the same name and must be unified: fold the deunicode step into
//! `CjkTokenizer::push_default_token` (the Latin path ONLY — never apply
//! deunicode to the Han bigram path, it would Pinyin-transliterate the
//! bigrams) and delete this module.

use deunicode::deunicode;
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Drop tokens whose UTF-8 byte length reaches this limit — the same limit
/// as tantivy's built-in default (`RemoveLongFilter::limit(40)` keeps
/// `len < 40`). Applied after deunicode (which can lengthen a token, e.g.
/// "æ" -> "ae").
const MAX_TOKEN_BYTES: usize = 40;

/// Latin-script tokenizer. See the module docs for the contract.
#[derive(Clone, Default)]
pub struct LatinTokenizer;

/// Token stream produced by [`LatinTokenizer`]. Tokens are computed eagerly;
/// positions increment once per emitted token.
#[derive(Clone, Default)]
pub struct LatinTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

impl Tokenizer for LatinTokenizer {
    type TokenStream<'a> = LatinTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> LatinTokenStream {
        let mut tokens = Vec::new();
        let mut position = 0usize;
        // Replicates SimpleTokenizer: maximal runs of alphanumeric chars.
        let mut word_start: Option<usize> = None;
        let mut flush = |tokens: &mut Vec<Token>,
                         text: &str,
                         start: usize,
                         end: usize,
                         position: &mut usize| {
            let word = &text[start..end];
            // ASCII-fold first (matches `normalize_for_index` ordering),
            // then lowercase — same as LowerCaser after folding.
            let folded = deunicode(word).to_lowercase();
            if folded.len() >= MAX_TOKEN_BYTES || folded.is_empty() {
                return;
            }
            tokens.push(Token {
                text: folded,
                offset_from: start,
                offset_to: end,
                position: *position,
                position_length: 1,
            });
            *position += 1;
        };
        for (idx, ch) in text.char_indices() {
            if ch.is_alphanumeric() {
                if word_start.is_none() {
                    word_start = Some(idx);
                }
            } else if let Some(start) = word_start.take() {
                flush(&mut tokens, text, start, idx, &mut position);
            }
        }
        if let Some(start) = word_start.take() {
            flush(&mut tokens, text, start, text.len(), &mut position);
        }
        LatinTokenStream { tokens, index: 0 }
    }
}

impl TokenStream for LatinTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

/// Register the Latin tokenizer as the `"default"` override on `index`.
/// Call once per index creation (and on open — the manager is per-Index,
/// so already-existing on-disk indexes pick it up too).
pub(crate) fn register_latin_tokenizer(index: &tantivy::Index) {
    index.tokenizers().register("default", LatinTokenizer);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_of(text: &str) -> Vec<String> {
        let mut t = LatinTokenizer;
        let mut stream = t.token_stream(text);
        let mut out = Vec::new();
        while stream.advance() {
            out.push(stream.token().text.clone());
        }
        out
    }

    #[test]
    fn folds_accents() {
        assert_eq!(tokens_of("El niño juega"), vec!["el", "nino", "juega"]);
        assert_eq!(tokens_of("¿Dónde está?"), vec!["donde", "esta"]);
    }

    #[test]
    fn ascii_unchanged() {
        assert_eq!(
            tokens_of("The quick brown fox"),
            vec!["the", "quick", "brown", "fox"]
        );
    }

    #[test]
    fn drops_long_tokens() {
        let long = "a".repeat(39);
        assert_eq!(tokens_of(&long), vec![long]);
        let too_long = "a".repeat(40);
        assert!(tokens_of(&too_long).is_empty());
    }
}
