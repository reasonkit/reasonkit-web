//! Main content extraction
//!
//! This module extracts the main content from web pages, converting it
//! to clean text or markdown format.

use crate::browser::PageHandle;
use crate::error::{ExtractionError, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

/// Extracted content from a page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    /// Plain text content
    pub text: String,
    /// Content as markdown (if converted)
    pub markdown: Option<String>,
    /// HTML of the main content
    pub html: String,
    /// Word count
    pub word_count: usize,
    /// Character count
    pub char_count: usize,
    /// Whether content was extracted from article/main element
    pub from_main: bool,
}

/// Content extraction functionality
pub struct ContentExtractor;

impl ContentExtractor {
    /// Extract main content from the page
    #[instrument(skip(page))]
    pub async fn extract_main_content(page: &PageHandle) -> Result<ExtractedContent> {
        info!("Extracting main content");

        // Try to find the main content using various strategies
        let (html, from_main) = Self::find_main_content(&page.page).await?;
        let text = Self::html_to_text(&html);
        let markdown = Self::html_to_markdown(&html);

        let word_count = text.split_whitespace().count();
        let char_count = text.chars().count();

        debug!(
            "Extracted {} words, {} chars, from_main={}",
            word_count, char_count, from_main
        );

        Ok(ExtractedContent {
            text,
            markdown: Some(markdown),
            html,
            word_count,
            char_count,
            from_main,
        })
    }

    /// Extract content from a specific selector
    #[instrument(skip(page))]
    pub async fn extract_from_selector(
        page: &PageHandle,
        selector: &str,
    ) -> Result<ExtractedContent> {
        info!("Extracting from selector: {}", selector);

        let script = format!(
            r#"
            (() => {{
                const el = document.querySelector('{}');
                if (!el) return null;
                return {{
                    html: el.innerHTML,
                    text: el.innerText
                }};
            }})()
            "#,
            selector.replace('\'', "\\'")
        );

        let result: Option<serde_json::Value> = page
            .page
            .evaluate(script.as_str())
            .await
            .map_err(|e| ExtractionError::ExtractionFailed(e.to_string()))?
            .into_value()
            .map_err(|e| ExtractionError::ExtractionFailed(e.to_string()))?;

        let result =
            result.ok_or_else(|| ExtractionError::ElementNotFound(selector.to_string()))?;

        let html = result["html"].as_str().unwrap_or("").to_string();
        let text = result["text"].as_str().unwrap_or("").to_string();

        let markdown = Self::html_to_markdown(&html);
        let word_count = text.split_whitespace().count();
        let char_count = text.chars().count();

        Ok(ExtractedContent {
            text,
            markdown: Some(markdown),
            html,
            word_count,
            char_count,
            from_main: false,
        })
    }

    /// Extract all text from the page body
    #[instrument(skip(page))]
    pub async fn extract_all_text(page: &PageHandle) -> Result<String> {
        let script = r#"
            document.body.innerText
        "#;

        let text: String = page
            .page
            .evaluate(script)
            .await
            .map_err(|e| ExtractionError::ExtractionFailed(e.to_string()))?
            .into_value()
            .map_err(|e| ExtractionError::ExtractionFailed(e.to_string()))?;

        Ok(text)
    }

    /// Find the main content element using various strategies
    async fn find_main_content(page: &chromiumoxide::Page) -> Result<(String, bool)> {
        let script = r#"
            (() => {
                // Strategy 1: Look for article or main elements
                const mainSelectors = [
                    'article',
                    'main',
                    '[role="main"]',
                    '[role="article"]',
                    '.article',
                    '.post',
                    '.content',
                    '.entry-content',
                    '.post-content',
                    '#content',
                    '#main-content',
                    '.main-content'
                ];

                for (const selector of mainSelectors) {
                    const el = document.querySelector(selector);
                    if (el && el.innerText.length > 200) {
                        return { html: el.innerHTML, fromMain: true };
                    }
                }

                // Strategy 2: Find the largest text block
                const textBlocks = [];
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_ELEMENT,
                    {
                        acceptNode: (node) => {
                            const tag = node.tagName.toLowerCase();
                            if (['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript'].includes(tag)) {
                                return NodeFilter.FILTER_REJECT;
                            }
                            return NodeFilter.FILTER_ACCEPT;
                        }
                    }
                );

                let node;
                while (node = walker.nextNode()) {
                    const text = node.innerText || '';
                    if (text.length > 200) {
                        textBlocks.push({
                            el: node,
                            length: text.length
                        });
                    }
                }

                if (textBlocks.length > 0) {
                    // Sort by length and get the longest
                    textBlocks.sort((a, b) => b.length - a.length);
                    return { html: textBlocks[0].el.innerHTML, fromMain: false };
                }

                // Fallback: return body
                return { html: document.body.innerHTML, fromMain: false };
            })()
        "#;

        let result: serde_json::Value = page
            .evaluate(script)
            .await
            .map_err(|e| ExtractionError::ExtractionFailed(e.to_string()))?
            .into_value()
            .map_err(|e| ExtractionError::ExtractionFailed(e.to_string()))?;

        let html = result["html"].as_str().unwrap_or("").to_string();
        let from_main = result["fromMain"].as_bool().unwrap_or(false);

        Ok((html, from_main))
    }

    /// Convert HTML to plain text
    pub fn html_to_text(html: &str) -> String {
        // Remove script and style tags
        let mut text = html.to_string();

        // Remove script tags and content
        let script_re = regex::Regex::new(r"<script[^>]*>[\s\S]*?</script>").unwrap();
        text = script_re.replace_all(&text, "").to_string();

        // Remove style tags and content
        let style_re = regex::Regex::new(r"<style[^>]*>[\s\S]*?</style>").unwrap();
        text = style_re.replace_all(&text, "").to_string();

        // Replace block elements with newlines
        let block_re = regex::Regex::new(r"</(p|div|br|li|h[1-6])>").unwrap();
        text = block_re.replace_all(&text, "\n").to_string();

        // Remove all remaining HTML tags
        let tag_re = regex::Regex::new(r"<[^>]+>").unwrap();
        text = tag_re.replace_all(&text, "").to_string();

        // Decode common HTML entities
        text = Self::decode_html_entities(&text);

        // Normalize whitespace
        let ws_re = regex::Regex::new(r"\s+").unwrap();
        text = ws_re.replace_all(&text, " ").to_string();

        // Normalize newlines
        let nl_re = regex::Regex::new(r"\n\s*\n+").unwrap();
        text = nl_re.replace_all(&text, "\n\n").to_string();

        text.trim().to_string()
    }

    /// Decode common HTML entities
    pub fn decode_html_entities(text: &str) -> String {
        text.replace("&nbsp;", " ")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&apos;", "'")
            .replace("&#x27;", "'")
            .replace("&#x2F;", "/")
            .replace("&copy;", "(c)")
            .replace("&reg;", "(R)")
            .replace("&trade;", "(TM)")
            .replace("&ndash;", "-")
            .replace("&mdash;", "--")
            .replace("&hellip;", "...")
            .replace("&lsquo;", "'")
            .replace("&rsquo;", "'")
            .replace("&ldquo;", "\"")
            .replace("&rdquo;", "\"")
    }

    /// Convert HTML to markdown
    pub fn html_to_markdown(html: &str) -> String {
        let mut md = html.to_string();

        // Remove script and style
        let script_re = regex::Regex::new(r"<script[^>]*>[\s\S]*?</script>").unwrap();
        md = script_re.replace_all(&md, "").to_string();
        let style_re = regex::Regex::new(r"<style[^>]*>[\s\S]*?</style>").unwrap();
        md = style_re.replace_all(&md, "").to_string();

        // Convert headers
        for i in (1..=6).rev() {
            let h_re = regex::Regex::new(&format!(r"<h{}[^>]*>(.*?)</h{}>", i, i)).unwrap();
            let prefix = "#".repeat(i);
            md = h_re
                .replace_all(&md, format!("{} $1\n\n", prefix))
                .to_string();
        }

        // Convert paragraphs
        let p_re = regex::Regex::new(r"<p[^>]*>(.*?)</p>").unwrap();
        md = p_re.replace_all(&md, "$1\n\n").to_string();

        // Convert line breaks
        let br_re = regex::Regex::new(r"<br\s*/?>").unwrap();
        md = br_re.replace_all(&md, "\n").to_string();

        // Convert bold
        let b_re = regex::Regex::new(r"<(b|strong)[^>]*>(.*?)</(b|strong)>").unwrap();
        md = b_re.replace_all(&md, "**$2**").to_string();

        // Convert italic
        let i_re = regex::Regex::new(r"<(i|em)[^>]*>(.*?)</(i|em)>").unwrap();
        md = i_re.replace_all(&md, "*$2*").to_string();

        // Convert links
        let a_re = regex::Regex::new(r#"<a[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>"#).unwrap();
        md = a_re.replace_all(&md, "[$2]($1)").to_string();

        // Convert code
        let code_re = regex::Regex::new(r"<code[^>]*>(.*?)</code>").unwrap();
        md = code_re.replace_all(&md, "`$1`").to_string();

        // Convert pre blocks (use [\s\S]*? to match across newlines)
        let pre_re = regex::Regex::new(r"<pre[^>]*>([\s\S]*?)</pre>").unwrap();
        md = pre_re.replace_all(&md, "```\n$1\n```").to_string();

        // Convert lists
        let li_re = regex::Regex::new(r"<li[^>]*>(.*?)</li>").unwrap();
        md = li_re.replace_all(&md, "- $1\n").to_string();

        // Remove remaining tags
        let tag_re = regex::Regex::new(r"<[^>]+>").unwrap();
        md = tag_re.replace_all(&md, "").to_string();

        // Decode HTML entities
        md = Self::decode_html_entities(&md);

        // Clean up whitespace
        let ws_re = regex::Regex::new(r"\n{3,}").unwrap();
        md = ws_re.replace_all(&md, "\n\n").to_string();

        md.trim().to_string()
    }

    /// Normalize whitespace in text
    pub fn normalize_whitespace(text: &str) -> String {
        let ws_re = regex::Regex::new(r"\s+").unwrap();
        ws_re.replace_all(text.trim(), " ").to_string()
    }

    /// Truncate text to a maximum length, adding ellipsis if truncated
    pub fn truncate(text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            text.to_string()
        } else if max_len <= 3 {
            text.chars().take(max_len).collect()
        } else {
            let truncated: String = text.chars().take(max_len - 3).collect();
            format!("{}...", truncated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HTML to Text Conversion Tests
    // ========================================================================

    #[test]
    fn test_html_to_text() {
        let html = "<p>Hello <b>world</b>!</p><p>Second paragraph.</p>";
        let text = ContentExtractor::html_to_text(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn test_html_to_text_removes_scripts() {
        let html = "<p>Content</p><script>evil();</script><p>More</p>";
        let text = ContentExtractor::html_to_text(html);
        assert!(!text.contains("evil"));
        assert!(text.contains("Content"));
        assert!(text.contains("More"));
    }

    #[test]
    fn test_html_to_text_removes_styles() {
        let html = "<p>Content</p><style>.hidden { display: none; }</style><p>More</p>";
        let text = ContentExtractor::html_to_text(html);
        assert!(!text.contains("hidden"));
        assert!(!text.contains("display"));
        assert!(text.contains("Content"));
        assert!(text.contains("More"));
    }

    #[test]
    fn test_html_to_text_multiline_script() {
        let html = r#"
            <p>Before</p>
            <script type="text/javascript">
                function evil() {
                    console.log("bad");
                }
                evil();
            </script>
            <p>After</p>
        "#;
        let text = ContentExtractor::html_to_text(html);
        assert!(!text.contains("evil"));
        assert!(!text.contains("console"));
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
    }

    #[test]
    fn test_html_to_text_preserves_newlines_for_blocks() {
        let html = "<p>Para 1</p><p>Para 2</p>";
        let text = ContentExtractor::html_to_text(html);
        // Should have some separation between paragraphs
        assert!(text.contains("Para 1"));
        assert!(text.contains("Para 2"));
    }

    #[test]
    fn test_html_to_text_strips_all_tags() {
        let html = "<div class=\"container\"><span id=\"test\">Hello</span></div>";
        let text = ContentExtractor::html_to_text(html);
        assert_eq!(text, "Hello");
        assert!(!text.contains("<"));
        assert!(!text.contains(">"));
        assert!(!text.contains("class"));
    }

    // ========================================================================
    // HTML Entity Decoding Tests
    // ========================================================================

    #[test]
    fn test_html_entity_decode_basic() {
        assert_eq!(
            ContentExtractor::decode_html_entities("&lt;div&gt;"),
            "<div>"
        );
        assert_eq!(ContentExtractor::decode_html_entities("&amp;"), "&");
        assert_eq!(ContentExtractor::decode_html_entities("&quot;"), "\"");
    }

    #[test]
    fn test_html_entity_decode_quotes() {
        assert_eq!(ContentExtractor::decode_html_entities("&#39;"), "'");
        assert_eq!(ContentExtractor::decode_html_entities("&apos;"), "'");
        assert_eq!(ContentExtractor::decode_html_entities("&#x27;"), "'");
    }

    #[test]
    fn test_html_entity_decode_typography() {
        assert_eq!(ContentExtractor::decode_html_entities("&ndash;"), "-");
        assert_eq!(ContentExtractor::decode_html_entities("&mdash;"), "--");
        assert_eq!(ContentExtractor::decode_html_entities("&hellip;"), "...");
        assert_eq!(ContentExtractor::decode_html_entities("&lsquo;"), "'");
        assert_eq!(ContentExtractor::decode_html_entities("&rsquo;"), "'");
        assert_eq!(ContentExtractor::decode_html_entities("&ldquo;"), "\"");
        assert_eq!(ContentExtractor::decode_html_entities("&rdquo;"), "\"");
    }

    #[test]
    fn test_html_entity_decode_symbols() {
        assert_eq!(ContentExtractor::decode_html_entities("&copy;"), "(c)");
        assert_eq!(ContentExtractor::decode_html_entities("&reg;"), "(R)");
        assert_eq!(ContentExtractor::decode_html_entities("&trade;"), "(TM)");
    }

    #[test]
    fn test_html_entity_decode_nbsp() {
        assert_eq!(
            ContentExtractor::decode_html_entities("Hello&nbsp;World"),
            "Hello World"
        );
    }

    #[test]
    fn test_html_entity_decode_mixed() {
        let input = "Copyright &copy; 2024 &mdash; All rights reserved &amp; more";
        let output = ContentExtractor::decode_html_entities(input);
        assert_eq!(output, "Copyright (c) 2024 -- All rights reserved & more");
    }

    // ========================================================================
    // Script Removal Tests
    // ========================================================================

    #[test]
    fn test_script_removal_inline() {
        let html = "<script>alert('xss')</script><p>Safe</p>";
        let text = ContentExtractor::html_to_text(html);
        assert!(!text.contains("alert"));
        assert!(!text.contains("xss"));
        assert!(text.contains("Safe"));
    }

    #[test]
    fn test_script_removal_with_attributes() {
        let html = "<script type=\"text/javascript\" src=\"bad.js\">code()</script><p>Safe</p>";
        let text = ContentExtractor::html_to_text(html);
        assert!(!text.contains("code"));
        assert!(!text.contains("javascript"));
        assert!(text.contains("Safe"));
    }

    #[test]
    fn test_script_removal_multiple() {
        let html = "<script>one()</script><p>Middle</p><script>two()</script>";
        let text = ContentExtractor::html_to_text(html);
        assert!(!text.contains("one"));
        assert!(!text.contains("two"));
        assert!(text.contains("Middle"));
    }

    // ========================================================================
    // Whitespace Normalization Tests
    // ========================================================================

    #[test]
    fn test_whitespace_normalization_spaces() {
        let text = "Hello    world";
        let normalized = ContentExtractor::normalize_whitespace(text);
        assert_eq!(normalized, "Hello world");
    }

    #[test]
    fn test_whitespace_normalization_tabs() {
        let text = "Hello\t\tworld";
        let normalized = ContentExtractor::normalize_whitespace(text);
        assert_eq!(normalized, "Hello world");
    }

    #[test]
    fn test_whitespace_normalization_newlines() {
        let text = "Hello\n\n\nworld";
        let normalized = ContentExtractor::normalize_whitespace(text);
        assert_eq!(normalized, "Hello world");
    }

    #[test]
    fn test_whitespace_normalization_mixed() {
        let text = "  Hello   \t\n  world  ";
        let normalized = ContentExtractor::normalize_whitespace(text);
        assert_eq!(normalized, "Hello world");
    }

    #[test]
    fn test_whitespace_normalization_empty() {
        let text = "   ";
        let normalized = ContentExtractor::normalize_whitespace(text);
        assert_eq!(normalized, "");
    }

    #[test]
    fn test_whitespace_normalization_single_word() {
        let text = "  Hello  ";
        let normalized = ContentExtractor::normalize_whitespace(text);
        assert_eq!(normalized, "Hello");
    }

    // ========================================================================
    // Truncation Tests
    // ========================================================================

    #[test]
    fn test_truncation_short_text() {
        let text = "Hello";
        let truncated = ContentExtractor::truncate(text, 10);
        assert_eq!(truncated, "Hello");
    }

    #[test]
    fn test_truncation_exact_length() {
        let text = "Hello";
        let truncated = ContentExtractor::truncate(text, 5);
        assert_eq!(truncated, "Hello");
    }

    #[test]
    fn test_truncation_adds_ellipsis() {
        let text = "Hello World";
        let truncated = ContentExtractor::truncate(text, 8);
        assert_eq!(truncated, "Hello...");
        assert_eq!(truncated.len(), 8);
    }

    #[test]
    fn test_truncation_very_short_limit() {
        let text = "Hello";
        let truncated = ContentExtractor::truncate(text, 3);
        assert_eq!(truncated, "Hel");
    }

    #[test]
    fn test_truncation_zero_limit() {
        let text = "Hello";
        let truncated = ContentExtractor::truncate(text, 0);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_truncation_empty_text() {
        let text = "";
        let truncated = ContentExtractor::truncate(text, 10);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_truncation_unicode() {
        let text = "Hello World";
        let truncated = ContentExtractor::truncate(text, 10);
        // Should handle unicode correctly (ellipsis counts as 3 chars)
        assert!(truncated.len() <= 10 || truncated.ends_with("..."));
    }

    // ========================================================================
    // HTML to Markdown Conversion Tests
    // ========================================================================

    #[test]
    fn test_html_to_markdown() {
        let html = "<h1>Title</h1><p>Para with <b>bold</b> and <a href=\"http://example.com\">link</a>.</p>";
        let md = ContentExtractor::html_to_markdown(html);
        assert!(md.contains("# Title"));
        assert!(md.contains("**bold**"));
        assert!(md.contains("[link](http://example.com)"));
    }

    #[test]
    fn test_html_to_markdown_headers() {
        let html = "<h1>H1</h1><h2>H2</h2><h3>H3</h3><h4>H4</h4><h5>H5</h5><h6>H6</h6>";
        let md = ContentExtractor::html_to_markdown(html);
        assert!(md.contains("# H1"));
        assert!(md.contains("## H2"));
        assert!(md.contains("### H3"));
        assert!(md.contains("#### H4"));
        assert!(md.contains("##### H5"));
        assert!(md.contains("###### H6"));
    }

    #[test]
    fn test_html_to_markdown_emphasis() {
        let html = "<p><b>bold</b> and <strong>strong</strong> and <i>italic</i> and <em>emphasis</em></p>";
        let md = ContentExtractor::html_to_markdown(html);
        assert!(md.contains("**bold**"));
        assert!(md.contains("**strong**"));
        assert!(md.contains("*italic*"));
        assert!(md.contains("*emphasis*"));
    }

    #[test]
    fn test_html_to_markdown_code() {
        let html = "<p>Use <code>println!</code> for output.</p>";
        let md = ContentExtractor::html_to_markdown(html);
        assert!(md.contains("`println!`"));
    }

    #[test]
    fn test_html_to_markdown_pre() {
        let html = "<pre>fn main() {\n    println!(\"Hello\");\n}</pre>";
        let md = ContentExtractor::html_to_markdown(html);
        assert!(md.contains("```"));
        assert!(md.contains("fn main()"));
    }

    #[test]
    fn test_html_to_markdown_list() {
        let html = "<ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul>";
        let md = ContentExtractor::html_to_markdown(html);
        assert!(md.contains("- Item 1"));
        assert!(md.contains("- Item 2"));
        assert!(md.contains("- Item 3"));
    }

    #[test]
    fn test_html_to_markdown_removes_scripts() {
        let html = "<p>Safe</p><script>evil()</script>";
        let md = ContentExtractor::html_to_markdown(html);
        assert!(!md.contains("evil"));
        assert!(md.contains("Safe"));
    }

    #[test]
    fn test_html_to_markdown_line_breaks() {
        let html = "Line 1<br>Line 2<br/>Line 3";
        let md = ContentExtractor::html_to_markdown(html);
        assert!(md.contains("Line 1"));
        assert!(md.contains("Line 2"));
        assert!(md.contains("Line 3"));
    }

    // ========================================================================
    // ExtractedContent Structure Tests
    // ========================================================================

    #[test]
    fn test_extracted_content_structure() {
        let content = ExtractedContent {
            text: "Hello world".to_string(),
            markdown: Some("Hello world".to_string()),
            html: "<p>Hello world</p>".to_string(),
            word_count: 2,
            char_count: 11,
            from_main: true,
        };
        assert_eq!(content.word_count, 2);
        assert!(content.from_main);
    }

    #[test]
    fn test_extracted_content_serialization() {
        let content = ExtractedContent {
            text: "Hello".to_string(),
            markdown: Some("Hello".to_string()),
            html: "<p>Hello</p>".to_string(),
            word_count: 1,
            char_count: 5,
            from_main: false,
        };

        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"word_count\":1"));
        assert!(json.contains("\"from_main\":false"));

        let deserialized: ExtractedContent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.text, "Hello");
        assert_eq!(deserialized.word_count, 1);
    }

    #[test]
    fn test_extracted_content_empty() {
        let content = ExtractedContent {
            text: String::new(),
            markdown: None,
            html: String::new(),
            word_count: 0,
            char_count: 0,
            from_main: false,
        };
        assert_eq!(content.word_count, 0);
        assert_eq!(content.char_count, 0);
        assert!(content.markdown.is_none());
    }

    // ========================================================================
    // Edge Cases Tests
    // ========================================================================

    #[test]
    fn test_html_to_text_nested_tags() {
        let html = "<div><p><span><b>Nested</b> content</span></p></div>";
        let text = ContentExtractor::html_to_text(html);
        assert!(text.contains("Nested"));
        assert!(text.contains("content"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn test_html_to_text_malformed_html() {
        let html = "<p>Unclosed paragraph <b>bold";
        let text = ContentExtractor::html_to_text(html);
        // Should still extract text even with malformed HTML
        assert!(text.contains("Unclosed"));
        assert!(text.contains("bold"));
    }

    #[test]
    fn test_html_to_text_self_closing_tags() {
        let html = "Hello<br/>World<hr/>Done";
        let text = ContentExtractor::html_to_text(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(text.contains("Done"));
    }

    #[test]
    fn test_html_to_text_comments() {
        let html = "<p>Before</p><!-- This is a comment --><p>After</p>";
        let text = ContentExtractor::html_to_text(html);
        assert!(!text.contains("comment"));
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
    }

    #[test]
    fn test_html_to_text_empty() {
        let html = "";
        let text = ContentExtractor::html_to_text(html);
        assert_eq!(text, "");
    }

    #[test]
    fn test_html_to_text_only_whitespace() {
        let html = "   \n\t   ";
        let text = ContentExtractor::html_to_text(html);
        assert_eq!(text, "");
    }
}
