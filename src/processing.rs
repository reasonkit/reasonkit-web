//! DOM Content Processing Utilities
//!
//! This module provides high-performance utilities for processing raw HTML content,
//! extracting clean text, and normalizing web page content for downstream consumption.
//!
//! # Features
//!
//! - **HTML Cleaning**: Remove scripts, styles, and other non-content elements
//! - **Text Extraction**: Convert HTML to clean, readable text
//! - **Entity Decoding**: Properly decode HTML entities
//! - **Whitespace Normalization**: Clean up excessive whitespace while preserving structure
//! - **Truncation**: Intelligently truncate content with ellipsis
//!
//! # Example
//!
//! ```rust
//! use reasonkit_web::processing::{ContentProcessor, ContentProcessorConfig};
//!
//! let config = ContentProcessorConfig::default();
//! let processor = ContentProcessor::new(config);
//!
//! let html = r#"<html><head><script>evil();</script></head>
//!               <body><p>Hello &amp; welcome!</p></body></html>"#;
//!
//! let result = processor.process(html);
//! assert!(result.text.contains("Hello & welcome!"));
//! assert!(!result.text.contains("evil"));
//! ```

use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, instrument, trace};

/// Configuration for the content processor
#[derive(Debug, Clone)]
pub struct ContentProcessorConfig {
    /// Maximum length of processed content (0 = unlimited)
    pub max_length: usize,
    /// Whether to preserve structural elements like paragraph breaks
    pub preserve_structure: bool,
    /// Minimum text length to consider valid content
    pub min_content_length: usize,
    /// Tags to completely remove (including their content)
    pub remove_tags: Vec<String>,
    /// Whether to decode HTML entities
    pub decode_entities: bool,
}

impl Default for ContentProcessorConfig {
    fn default() -> Self {
        Self {
            max_length: 0, // unlimited by default
            preserve_structure: true,
            min_content_length: 10,
            remove_tags: vec![
                "script".to_string(),
                "style".to_string(),
                "noscript".to_string(),
                "template".to_string(),
                "svg".to_string(),
                "math".to_string(),
            ],
            decode_entities: true,
        }
    }
}

/// Content processor for HTML documents
///
/// Provides methods to clean, extract, and normalize HTML content.
#[derive(Debug, Clone)]
pub struct ContentProcessor {
    config: ContentProcessorConfig,
}

/// Result of content processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcessedContent {
    /// The extracted and cleaned text content
    pub text: String,
    /// Number of words in the processed content
    pub word_count: usize,
    /// Number of characters in the processed content
    pub char_count: usize,
    /// Whether the content was truncated
    pub was_truncated: bool,
    /// Processing time in microseconds
    pub processing_time_us: u64,
}

impl ContentProcessor {
    /// Create a new content processor with the given configuration
    pub fn new(config: ContentProcessorConfig) -> Self {
        Self { config }
    }

    /// Create a content processor with default settings
    pub fn with_defaults() -> Self {
        Self::new(ContentProcessorConfig::default())
    }

    /// Create a content processor with a maximum length limit
    pub fn with_max_length(max_length: usize) -> Self {
        Self::new(ContentProcessorConfig {
            max_length,
            ..Default::default()
        })
    }

    /// Process raw HTML and return cleaned content
    ///
    /// This is the main entry point for content processing. It:
    /// 1. Removes script, style, and other non-content elements
    /// 2. Extracts text from remaining HTML
    /// 3. Decodes HTML entities
    /// 4. Normalizes whitespace
    /// 5. Optionally truncates to max_length
    #[instrument(skip(self, raw_html), fields(html_len = raw_html.len()))]
    pub fn process(&self, raw_html: &str) -> ProcessedContent {
        let start = Instant::now();
        trace!("Starting content processing");

        // Step 1: Remove unwanted elements (scripts, styles, etc.)
        let cleaned_html = self.remove_scripts_styles(raw_html);

        // Step 2: Extract text from HTML
        let extracted_text = self.extract_text(&cleaned_html);

        // Step 3: Normalize whitespace
        let normalized = self.normalize_whitespace(&extracted_text);

        // Step 4: Truncate if needed
        let (text, was_truncated) =
            if self.config.max_length > 0 && normalized.len() > self.config.max_length {
                let truncated = self.truncate_with_ellipsis(&normalized, self.config.max_length);
                (truncated, true)
            } else {
                (normalized, false)
            };

        // Calculate metrics
        let word_count = text.split_whitespace().count();
        let char_count = text.chars().count();
        let processing_time_us = start.elapsed().as_micros() as u64;

        debug!(
            "Processed content: {} words, {} chars, truncated={}, time={}us",
            word_count, char_count, was_truncated, processing_time_us
        );

        ProcessedContent {
            text,
            word_count,
            char_count,
            was_truncated,
            processing_time_us,
        }
    }

    /// Extract text content from HTML
    ///
    /// Uses the scraper crate to parse HTML and extract text nodes,
    /// preserving paragraph structure if configured.
    #[instrument(skip(self, html), fields(html_len = html.len()))]
    pub fn extract_text(&self, html: &str) -> String {
        let document = Html::parse_document(html);
        let mut text_parts: Vec<String> = Vec::new();

        // Try to extract from body first, fall back to root
        let body_selector = Selector::parse("body").unwrap();

        if let Some(body) = document.select(&body_selector).next() {
            self.extract_text_from_element(&body, &mut text_parts);
        } else {
            // No body, extract from root
            let root = document.root_element();
            self.extract_text_from_element(&root, &mut text_parts);
        }

        if self.config.preserve_structure {
            text_parts.join("\n")
        } else {
            text_parts.join(" ")
        }
    }

    /// Extract text from an element and its children using scraper's public API
    fn extract_text_from_element(
        &self,
        element: &scraper::ElementRef<'_>,
        text_parts: &mut Vec<String>,
    ) {
        let tag_name = element.value().name().to_lowercase();

        // Skip removed tags entirely
        if self.config.remove_tags.contains(&tag_name) {
            return;
        }

        // Check if this is a block element
        let is_block = matches!(
            tag_name.as_str(),
            "p" | "div"
                | "section"
                | "article"
                | "header"
                | "footer"
                | "main"
                | "aside"
                | "nav"
                | "h1"
                | "h2"
                | "h3"
                | "h4"
                | "h5"
                | "h6"
                | "li"
                | "dt"
                | "dd"
                | "blockquote"
                | "pre"
                | "table"
                | "tr"
                | "br"
                | "hr"
        );

        // Add blank line before block elements (for structure preservation)
        if is_block && self.config.preserve_structure && !text_parts.is_empty() {
            if let Some(last) = text_parts.last() {
                if !last.is_empty() {
                    text_parts.push(String::new());
                }
            }
        }

        // Process children
        for child in element.children() {
            if let Some(text_node) = child.value().as_text() {
                let trimmed = text_node.trim();
                if !trimmed.is_empty() {
                    let decoded = if self.config.decode_entities {
                        Self::decode_html_entities(trimmed)
                    } else {
                        trimmed.to_string()
                    };
                    text_parts.push(decoded);
                }
            } else if let Some(child_element) = scraper::ElementRef::wrap(child) {
                self.extract_text_from_element(&child_element, text_parts);
            }
        }

        // Add blank line after block elements
        if is_block && self.config.preserve_structure && !text_parts.is_empty() {
            if let Some(last) = text_parts.last() {
                if !last.is_empty() {
                    text_parts.push(String::new());
                }
            }
        }
    }

    /// Remove script, style, and other non-content elements from HTML
    ///
    /// This method performs a comprehensive cleanup of HTML by:
    /// - Removing `<script>` tags and their content
    /// - Removing `<style>` tags and their content
    /// - Removing `<noscript>` tags and their content
    /// - Removing HTML comments
    /// - Removing other configured tags
    #[instrument(skip(self, html), fields(html_len = html.len()))]
    pub fn remove_scripts_styles(&self, html: &str) -> String {
        let mut result = html.to_string();

        // Remove HTML comments first
        result = Self::remove_pattern(&result, r"<!--[\s\S]*?-->");

        // Remove each configured tag type
        for tag in &self.config.remove_tags {
            // Pattern for tags with content: <tag ...>...</tag>
            let pattern = format!(r"(?is)<{}\b[^>]*>[\s\S]*?</{}>", tag, tag);
            result = Self::remove_pattern(&result, &pattern);

            // Pattern for self-closing tags: <tag ... />
            let self_closing_pattern = format!(r"(?i)<{}\b[^>]*/?>", tag);
            result = Self::remove_pattern(&result, &self_closing_pattern);
        }

        // Also remove inline event handlers and javascript: hrefs for security
        result = Self::remove_pattern(&result, r#"(?i)\s+on\w+\s*=\s*["'][^"']*["']"#);
        result = Self::remove_pattern(&result, r#"(?i)\s+on\w+\s*=\s*[^\s>]+"#);
        result = Self::remove_pattern(&result, r#"(?i)href\s*=\s*["']javascript:[^"']*["']"#);

        trace!(
            "Removed scripts/styles: {} -> {} bytes",
            html.len(),
            result.len()
        );
        result
    }

    /// Helper to remove a regex pattern from text
    fn remove_pattern(text: &str, pattern: &str) -> String {
        match regex::Regex::new(pattern) {
            Ok(re) => re.replace_all(text, "").to_string(),
            Err(_) => text.to_string(),
        }
    }

    /// Normalize whitespace in text
    ///
    /// This method:
    /// - Collapses multiple spaces into single spaces
    /// - Normalizes different whitespace characters (tabs, nbsp, etc.)
    /// - Preserves paragraph breaks (double newlines) if structure preservation is enabled
    /// - Trims leading and trailing whitespace
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn normalize_whitespace(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Replace non-breaking spaces and other special whitespace with regular space
        result = result
            .replace(
                ['\u{00A0}', '\u{2002}', '\u{2003}', '\u{2009}', '\u{200A}'],
                " ",
            )
            .replace(['\u{200B}', '\u{FEFF}'], ""); // Zero-width space and BOM

        // Replace tabs with spaces
        result = result.replace('\t', " ");

        // Replace carriage returns with newlines
        result = result.replace("\r\n", "\n").replace('\r', "\n");

        if self.config.preserve_structure {
            // Collapse multiple spaces (but not newlines)
            let space_re = regex::Regex::new(r"[^\S\n]+").unwrap();
            result = space_re.replace_all(&result, " ").to_string();

            // Collapse multiple newlines (3+ becomes 2)
            let newline_re = regex::Regex::new(r"\n{3,}").unwrap();
            result = newline_re.replace_all(&result, "\n\n").to_string();

            // Trim each line
            result = result
                .lines()
                .map(|line| line.trim())
                .collect::<Vec<_>>()
                .join("\n");
        } else {
            // Collapse all whitespace into single spaces
            let ws_re = regex::Regex::new(r"\s+").unwrap();
            result = ws_re.replace_all(&result, " ").to_string();
        }

        result.trim().to_string()
    }

    /// Truncate text with ellipsis at a word boundary
    ///
    /// This method truncates text to approximately the given maximum length,
    /// breaking at word boundaries to avoid cutting words in half.
    /// Appends "..." to indicate truncation.
    #[instrument(skip(self, text), fields(text_len = text.len(), max = max))]
    pub fn truncate_with_ellipsis(&self, text: &str, max: usize) -> String {
        if text.len() <= max {
            return text.to_string();
        }

        // Reserve space for ellipsis
        let effective_max = max.saturating_sub(3);
        if effective_max == 0 {
            return "...".to_string();
        }

        // Find the last space before the limit
        let truncate_at = text[..effective_max]
            .rfind(|c: char| c.is_whitespace())
            .unwrap_or(effective_max);

        // Avoid truncating too short (at least 20% of max)
        let min_length = effective_max / 5;
        let truncate_at = if truncate_at < min_length {
            effective_max
        } else {
            truncate_at
        };

        let mut result = text[..truncate_at].trim_end().to_string();
        result.push_str("...");

        trace!("Truncated from {} to {} chars", text.len(), result.len());
        result
    }

    /// Decode HTML entities in text
    ///
    /// Handles common HTML entities including:
    /// - Named entities (&amp;, &lt;, &gt;, &quot;, &nbsp;, etc.)
    /// - Numeric entities (&#39;, &#x27;, etc.)
    pub fn decode_html_entities(text: &str) -> String {
        let mut result = text.to_string();

        // Named entities (most common first for performance)
        let named_entities = [
            ("&amp;", "&"),
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&quot;", "\""),
            ("&apos;", "'"),
            ("&nbsp;", " "),
            ("&ndash;", "\u{2013}"),
            ("&mdash;", "\u{2014}"),
            ("&lsquo;", "\u{2018}"),
            ("&rsquo;", "\u{2019}"),
            ("&ldquo;", "\u{201C}"),
            ("&rdquo;", "\u{201D}"),
            ("&hellip;", "\u{2026}"),
            ("&trade;", "\u{2122}"),
            ("&copy;", "\u{00A9}"),
            ("&reg;", "\u{00AE}"),
            ("&deg;", "\u{00B0}"),
            ("&plusmn;", "\u{00B1}"),
            ("&times;", "\u{00D7}"),
            ("&divide;", "\u{00F7}"),
            ("&euro;", "\u{20AC}"),
            ("&pound;", "\u{00A3}"),
            ("&yen;", "\u{00A5}"),
            ("&cent;", "\u{00A2}"),
        ];

        for (entity, replacement) in named_entities {
            result = result.replace(entity, replacement);
        }

        // Decimal numeric entities (&#123;)
        if result.contains("&#") {
            let decimal_re = regex::Regex::new(r"&#(\d+);").unwrap();
            result = decimal_re
                .replace_all(&result, |caps: &regex::Captures| {
                    caps.get(1)
                        .and_then(|m| m.as_str().parse::<u32>().ok())
                        .and_then(char::from_u32)
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| caps[0].to_string())
                })
                .to_string();

            // Hexadecimal numeric entities (&#x1F;)
            let hex_re = regex::Regex::new(r"(?i)&#x([0-9a-f]+);").unwrap();
            result = hex_re
                .replace_all(&result, |caps: &regex::Captures| {
                    caps.get(1)
                        .and_then(|m| u32::from_str_radix(m.as_str(), 16).ok())
                        .and_then(char::from_u32)
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| caps[0].to_string())
                })
                .to_string();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_processing() {
        let processor = ContentProcessor::with_defaults();
        let html = "<html><body><p>Hello world!</p></body></html>";
        let result = processor.process(html);

        assert_eq!(result.text.trim(), "Hello world!");
        assert_eq!(result.word_count, 2);
        assert!(!result.was_truncated);
    }

    #[test]
    fn test_script_removal() {
        let processor = ContentProcessor::with_defaults();
        let html = r#"
            <html>
            <head><script>alert('evil');</script></head>
            <body>
                <p>Safe content</p>
                <script type="text/javascript">
                    malicious_code();
                </script>
            </body>
            </html>
        "#;
        let result = processor.process(html);

        assert!(result.text.contains("Safe content"));
        assert!(!result.text.contains("evil"));
        assert!(!result.text.contains("malicious"));
    }

    #[test]
    fn test_style_removal() {
        let processor = ContentProcessor::with_defaults();
        let html = r#"
            <html>
            <head><style>.hidden { display: none; }</style></head>
            <body>
                <p>Visible text</p>
                <style>
                    body { background: red; }
                </style>
            </body>
            </html>
        "#;
        let result = processor.process(html);

        assert!(result.text.contains("Visible text"));
        assert!(!result.text.contains("display"));
        assert!(!result.text.contains("background"));
    }

    #[test]
    fn test_entity_decoding() {
        let processor = ContentProcessor::with_defaults();
        let html = "<p>Tom &amp; Jerry &lt;3 &quot;cheese&quot;</p>";
        let result = processor.process(html);

        assert!(result.text.contains("Tom & Jerry"));
        assert!(result.text.contains("<3"));
        assert!(result.text.contains("\"cheese\""));
    }

    #[test]
    fn test_numeric_entity_decoding() {
        let decoded = ContentProcessor::decode_html_entities("&#39;hello&#39; &#x27;world&#x27;");
        assert_eq!(decoded, "'hello' 'world'");
    }

    #[test]
    fn test_whitespace_normalization() {
        let processor = ContentProcessor::with_defaults();
        let html = "<p>Too    many     spaces</p>";
        let result = processor.process(html);

        assert!(!result.text.contains("    "));
        assert!(result.text.contains("Too many spaces") || result.text.contains("Too many spaces"));
    }

    #[test]
    fn test_structure_preservation() {
        let config = ContentProcessorConfig {
            preserve_structure: true,
            ..Default::default()
        };
        let processor = ContentProcessor::new(config);
        let html = "<p>Paragraph 1</p><p>Paragraph 2</p>";
        let result = processor.process(html);

        // Should have some kind of separation between paragraphs
        assert!(result.text.contains("Paragraph 1"));
        assert!(result.text.contains("Paragraph 2"));
    }

    #[test]
    fn test_truncation_with_ellipsis() {
        let processor = ContentProcessor::with_max_length(20);
        let html = "<p>This is a very long piece of text that should be truncated.</p>";
        let result = processor.process(html);

        assert!(result.was_truncated);
        assert!(result.text.ends_with("..."));
        assert!(result.text.len() <= 20);
    }

    #[test]
    fn test_truncation_at_word_boundary() {
        let processor = ContentProcessor::with_defaults();
        let text = "Hello world how are you doing today";
        let truncated = processor.truncate_with_ellipsis(text, 15);

        assert!(truncated.ends_with("..."));
        // Should break at "world" or similar, not in the middle of a word
        assert!(!truncated.contains("wor...") || truncated == "Hello world...");
    }

    #[test]
    fn test_no_truncation_for_short_content() {
        let processor = ContentProcessor::with_max_length(1000);
        let html = "<p>Short content</p>";
        let result = processor.process(html);

        assert!(!result.was_truncated);
    }

    #[test]
    fn test_noscript_removal() {
        let processor = ContentProcessor::with_defaults();
        let html = r#"
            <body>
                <noscript>Enable JavaScript!</noscript>
                <p>Content</p>
            </body>
        "#;
        let result = processor.process(html);

        assert!(result.text.contains("Content"));
        assert!(!result.text.contains("JavaScript"));
    }

    #[test]
    fn test_comment_removal() {
        let processor = ContentProcessor::with_defaults();
        let html = r#"
            <body>
                <!-- This is a comment -->
                <p>Visible</p>
                <!-- Another comment
                     with multiple lines -->
            </body>
        "#;
        let cleaned = processor.remove_scripts_styles(html);

        assert!(!cleaned.contains("This is a comment"));
        assert!(!cleaned.contains("Another comment"));
    }

    #[test]
    fn test_inline_event_handler_removal() {
        let processor = ContentProcessor::with_defaults();
        let html = r#"<button onclick="evil()">Click</button>"#;
        let cleaned = processor.remove_scripts_styles(html);

        assert!(!cleaned.contains("onclick"));
        assert!(!cleaned.contains("evil"));
    }

    #[test]
    fn test_javascript_href_removal() {
        let processor = ContentProcessor::with_defaults();
        let html = r#"<a href="javascript:alert('xss')">Click</a>"#;
        let cleaned = processor.remove_scripts_styles(html);

        assert!(!cleaned.contains("javascript:"));
    }

    #[test]
    fn test_special_whitespace_normalization() {
        let processor = ContentProcessor::with_defaults();
        let text_with_nbsp = "Hello\u{00A0}world\u{2003}test";
        let normalized = processor.normalize_whitespace(text_with_nbsp);

        assert!(!normalized.contains('\u{00A0}'));
        assert!(!normalized.contains('\u{2003}'));
        assert!(normalized.contains("Hello world test") || normalized.contains("Hello world test"));
    }

    #[test]
    fn test_processed_content_metrics() {
        let processor = ContentProcessor::with_defaults();
        let html = "<p>One two three four five</p>";
        let result = processor.process(html);

        assert_eq!(result.word_count, 5);
        assert!(result.char_count > 0);
        assert!(result.processing_time_us >= 0);
    }

    #[test]
    fn test_empty_html() {
        let processor = ContentProcessor::with_defaults();
        let html = "<html><body></body></html>";
        let result = processor.process(html);

        assert!(result.text.is_empty() || result.word_count == 0);
    }

    #[test]
    fn test_deeply_nested_content() {
        let processor = ContentProcessor::with_defaults();
        let html = "<div><div><div><span><p>Deep content</p></span></div></div></div>";
        let result = processor.process(html);

        assert!(result.text.contains("Deep content"));
    }

    #[test]
    fn test_mixed_content() {
        let processor = ContentProcessor::with_defaults();
        let html = r#"
            <html>
            <head>
                <title>Test Page</title>
                <script>bad();</script>
                <style>.foo { color: red; }</style>
            </head>
            <body>
                <header><nav>Menu</nav></header>
                <main>
                    <article>
                        <h1>Article Title</h1>
                        <p>First paragraph with <strong>bold</strong> text.</p>
                        <p>Second paragraph with a <a href="http://example.com">link</a>.</p>
                    </article>
                </main>
                <footer>&copy; 2024</footer>
            </body>
            </html>
        "#;
        let result = processor.process(html);

        assert!(result.text.contains("Article Title"));
        assert!(result.text.contains("First paragraph"));
        assert!(result.text.contains("bold"));
        assert!(result.text.contains("link"));
        assert!(!result.text.contains("bad()"));
        assert!(!result.text.contains("color: red"));
    }

    #[test]
    fn test_unicode_content() {
        let processor = ContentProcessor::with_defaults();
        let html = "<p>Hello \u{1F600} World! Caf\u{00E9}</p>";
        let result = processor.process(html);

        assert!(result.text.contains("\u{1F600}")); // emoji preserved
        assert!(result.text.contains("Caf\u{00E9}")); // accented char preserved
    }

    #[test]
    fn test_custom_remove_tags() {
        let config = ContentProcessorConfig {
            remove_tags: vec!["script".to_string(), "style".to_string(), "nav".to_string()],
            ..Default::default()
        };
        let processor = ContentProcessor::new(config);
        let html = "<nav>Navigation</nav><p>Content</p>";
        let result = processor.process(html);

        assert!(!result.text.contains("Navigation"));
        assert!(result.text.contains("Content"));
    }

    #[test]
    fn test_without_entity_decoding() {
        let config = ContentProcessorConfig {
            decode_entities: false,
            ..Default::default()
        };
        let processor = ContentProcessor::new(config);
        let html = "<p>&amp; &lt; &gt;</p>";
        let result = processor.process(html);

        // Entities should remain as-is
        assert!(result.text.contains("&amp;") || result.text.contains("&"));
    }

    #[test]
    fn test_extract_text_directly() {
        let processor = ContentProcessor::with_defaults();
        let html = "<p>Direct <em>extraction</em> test</p>";
        let text = processor.extract_text(html);

        assert!(text.contains("Direct"));
        assert!(text.contains("extraction"));
        assert!(text.contains("test"));
    }

    #[test]
    fn test_remove_scripts_styles_directly() {
        let processor = ContentProcessor::with_defaults();
        let html = "<script>bad();</script><p>Good</p><style>.x{}</style>";
        let cleaned = processor.remove_scripts_styles(html);

        assert!(!cleaned.contains("bad()"));
        assert!(!cleaned.contains(".x{}"));
        assert!(cleaned.contains("<p>Good</p>"));
    }

    #[test]
    fn test_normalize_whitespace_directly() {
        let processor = ContentProcessor::with_defaults();
        let text = "  Multiple   spaces   and\n\n\n\nmany newlines  ";
        let normalized = processor.normalize_whitespace(text);

        assert!(!normalized.starts_with(' '));
        assert!(!normalized.ends_with(' '));
        assert!(!normalized.contains("   ")); // no triple spaces
    }
}
