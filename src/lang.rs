//! Language detection and per-language defaults for lint-ai's
//! language-by-language internationalization.
//!
//! Detection is script-based for CJK (count Han vs Hangul vs Latin
//! characters, winner takes it) plus Spanish signals for Latin-script text
//! (distinctive characters like ñ/¿/¡ and function-word hits — script
//! counting alone cannot separate Spanish from English). Ambiguous or
//! script-free text defaults to English.
//! Tokenization itself is script-aware per run (see [`crate::tokenizer`]),
//! so mixed-language text is handled without needing a single winner —
//! detection is used where a per-text language decision is required
//! (spaCy model selection, normalization).

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Content language. `Auto` (the default) means "detect from the text".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    /// Auto-detect from script statistics (plus Spanish signals for Latin
    /// text). Default.
    #[default]
    #[value(name = "auto")]
    Auto,
    #[value(name = "en")]
    En,
    #[value(name = "zh")]
    Zh,
    #[value(name = "ko")]
    Ko,
    #[value(name = "es")]
    Es,
}

impl Lang {
    /// Resolve to a concrete language: `Auto` runs [`detect_lang`].
    pub fn resolve(self, text: &str) -> Lang {
        match self {
            Lang::Auto => detect_lang(text),
            lang => lang,
        }
    }
}

/// True for CJK Unified Ideographs and extensions (but not Hangul,
/// Hiragana, Katakana, or CJK punctuation — 。、「」！？ etc. are
/// separators, not word characters, and must not join bigrams).
pub fn is_han(ch: char) -> bool {
    matches!(ch,
        '\u{3400}'..='\u{4DBF}'   // Ext A
        | '\u{4E00}'..='\u{9FFF}'   // Unified Ideographs
        | '\u{F900}'..='\u{FAFF}'   // Compatibility Ideographs
        | '\u{20000}'..='\u{2A6DF}' // Ext B
        | '\u{2A6E0}'..='\u{2CEAF}' // Ext C
        | '\u{2CEB0}'..='\u{2EBEF}' // Ext D/E/F
        | '\u{30000}'..='\u{3134F}' // Ext G
        | '\u{31350}'..='\u{323AF}' // Ext H
        | '\u{2F800}'..='\u{2FA1D}' // Compatibility Supplement
    )
}

/// True for Hangul syllables and Jamo.
pub fn is_hangul(ch: char) -> bool {
    matches!(ch,
        '\u{1100}'..='\u{11FF}'   // Hangul Jamo
        | '\u{3130}'..='\u{318F}'   // Hangul Compatibility Jamo
        | '\u{A960}'..='\u{A97F}'   // Hangul Jamo Extended-A
        | '\u{AC00}'..='\u{D7AF}'   // Hangul Syllables
        | '\u{D7B0}'..='\u{D7FF}' // Hangul Jamo Extended-B
    )
}

/// Characters that are strong Spanish signals. The accented vowels + ñ/ü
/// occur in normal Spanish prose; ¿ and ¡ are unique to Spanish
/// orthography.
fn is_spanish_distinctive(ch: char) -> bool {
    matches!(ch,
        'ñ' | 'Ñ'
        | 'á' | 'é' | 'í' | 'ó' | 'ú' | 'ü'
        | 'Á' | 'É' | 'Í' | 'Ó' | 'Ú' | 'Ü'
        | '¿' | '¡'
    )
}

/// Common Spanish function words (lowercase). Used for detection only —
/// never as a stopword list (see `crate::tokenizer` for that).
const SPANISH_FUNCTION_WORDS: &[&str] = &[
    "el", "la", "los", "las", "del", "al", "de", "que", "en", "y", "e", "ni", "o", "u", "un",
    "una", "unos", "unas", "se", "no", "por", "para", "con", "sin", "sobre", "entre", "hasta",
    "desde", "pero", "porque", "cuando", "donde", "como", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "mi", "tu", "su", "sus", "me", "te", "nos", "les", "lo",
    "hay", "muy", "tan", "mas", "tambien", "son", "es", "fue", "fueron", "era", "eran",
    "estan", "estoy", "esta",
];

/// True when Latin-script `text` looks Spanish. Needs either distinctive
/// characters (accents/ñ/¿/¡) or several function-word hits — a single
/// "el" or "no" in English prose must not flip the language.
fn is_spanish(text: &str) -> bool {
    let distinctive = text.chars().filter(|c| is_spanish_distinctive(*c)).count();
    if distinctive >= 2 {
        return true;
    }
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    if words.len() < 4 {
        return false;
    }
    let hits = words
        .iter()
        .filter(|w| SPANISH_FUNCTION_WORDS.contains(&w.as_str()))
        .count();
    if distinctive >= 1 {
        hits >= 2
    } else {
        // No accents at all (informal typing): demand more evidence and a
        // minimum length so a long English text with a few "no"/"en" words
        // does not flip.
        hits >= 4 && words.len() >= 8
    }
}

/// Detect the language of `text` from script statistics (plus Spanish
/// signals for Latin text). Never returns [`Lang::Auto`]; ambiguous or
/// script-free text defaults to English.
pub fn detect_lang(text: &str) -> Lang {
    let mut han = 0u32;
    let mut hangul = 0u32;
    let mut latin = 0u32;
    for ch in text.chars() {
        if is_han(ch) {
            han += 1;
        } else if is_hangul(ch) {
            hangul += 1;
        } else if ch.is_ascii_alphabetic() {
            latin += 1;
        }
    }
    if han > hangul && han > latin {
        Lang::Zh
    } else if hangul > han && hangul > latin {
        Lang::Ko
    } else if is_spanish(text) {
        Lang::Es
    } else {
        Lang::En
    }
}

/// The spaCy model to use for `lang` when the user did not pass an explicit
/// `--spacy-model`.
pub fn default_spacy_model_for_lang(lang: Lang) -> &'static str {
    match lang {
        Lang::Zh => "zh_core_web_sm",
        Lang::Es => "es_core_news_sm",
        _ => "en_core_web_sm",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_chinese() {
        assert_eq!(detect_lang("我毕业于清华大学"), Lang::Zh);
        assert_eq!(detect_lang("今天天气如何"), Lang::Zh);
    }

    #[test]
    fn detects_korean() {
        assert_eq!(detect_lang("한국어 테스트입니다"), Lang::Ko);
    }

    #[test]
    fn detects_english_and_defaults_on_ambiguous() {
        assert_eq!(detect_lang("What degree did I graduate with?"), Lang::En);
        assert_eq!(detect_lang(""), Lang::En);
        assert_eq!(detect_lang("12345 !!!"), Lang::En);
        // Tie between Han and Latin: ambiguous -> English.
        assert_eq!(detect_lang("中文ab"), Lang::En);
    }

    #[test]
    fn detects_spanish_with_accents() {
        assert_eq!(detect_lang("¿Dónde está la biblioteca?"), Lang::Es);
        assert_eq!(detect_lang("El niño juega en el parque"), Lang::Es);
        assert_eq!(
            detect_lang("Me gradué de la Universidad de Madrid en 2020"),
            Lang::Es
        );
    }

    #[test]
    fn detects_spanish_without_accents() {
        // Informal typing without accents: function words carry it.
        assert_eq!(
            detect_lang("Donde esta la biblioteca que busco desde ayer"),
            Lang::Es
        );
    }

    #[test]
    fn does_not_misfire_on_english() {
        assert_eq!(detect_lang("The cat is on the table"), Lang::En);
        assert_eq!(detect_lang("I have no idea where it is"), Lang::En);
        // Single Spanish-looking word is not enough.
        assert_eq!(detect_lang("hola"), Lang::En);
        assert_eq!(detect_lang("el"), Lang::En);
    }

    #[test]
    fn auto_resolves() {
        assert_eq!(detect_lang("¿Dónde está?"), Lang::Es);
        assert_eq!(Lang::Auto.resolve("¿Dónde está?"), Lang::Es);
        assert_eq!(Lang::Es.resolve("hello"), Lang::Es);
    }

    #[test]
    fn spacy_model_defaults() {
        assert_eq!(default_spacy_model_for_lang(Lang::Zh), "zh_core_web_sm");
        assert_eq!(default_spacy_model_for_lang(Lang::Es), "es_core_news_sm");
        assert_eq!(default_spacy_model_for_lang(Lang::En), "en_core_web_sm");
        assert_eq!(default_spacy_model_for_lang(Lang::Ko), "en_core_web_sm");
    }
}
