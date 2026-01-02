//! Page metadata extraction
//!
//! This module extracts page metadata including title, description,
//! Open Graph data, Twitter cards, and other structured data.

use crate::browser::PageHandle;
use crate::error::{ExtractionError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, instrument};

/// Extracted page metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageMetadata {
    /// Page title
    pub title: Option<String>,
    /// Meta description
    pub description: Option<String>,
    /// Canonical URL
    pub canonical: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// Open Graph metadata
    pub open_graph: OpenGraphData,
    /// Twitter Card metadata
    pub twitter_card: TwitterCardData,
    /// Favicon URL
    pub favicon: Option<String>,
    /// All meta tags
    pub meta_tags: HashMap<String, String>,
    /// JSON-LD structured data
    pub json_ld: Vec<serde_json::Value>,
}

/// Open Graph metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenGraphData {
    /// og:title
    pub title: Option<String>,
    /// og:description
    pub description: Option<String>,
    /// og:image
    pub image: Option<String>,
    /// og:url
    pub url: Option<String>,
    /// og:type
    pub og_type: Option<String>,
    /// og:site_name
    pub site_name: Option<String>,
    /// og:locale
    pub locale: Option<String>,
}

/// Twitter Card metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TwitterCardData {
    /// twitter:card
    pub card: Option<String>,
    /// twitter:title
    pub title: Option<String>,
    /// twitter:description
    pub description: Option<String>,
    /// twitter:image
    pub image: Option<String>,
    /// twitter:site
    pub site: Option<String>,
    /// twitter:creator
    pub creator: Option<String>,
}

/// Metadata extraction functionality
pub struct MetadataExtractor;

impl MetadataExtractor {
    /// Extract all metadata from the page
    #[instrument(skip(page))]
    pub async fn extract(page: &PageHandle) -> Result<PageMetadata> {
        info!("Extracting page metadata");

        let script = r#"
            (() => {
                const result = {
                    title: document.title,
                    description: null,
                    canonical: null,
                    language: document.documentElement.lang || null,
                    author: null,
                    keywords: [],
                    openGraph: {},
                    twitterCard: {},
                    favicon: null,
                    metaTags: {},
                    jsonLd: []
                };

                // Extract meta tags
                document.querySelectorAll('meta').forEach(meta => {
                    const name = meta.getAttribute('name') || meta.getAttribute('property');
                    const content = meta.getAttribute('content');

                    if (!name || !content) return;

                    result.metaTags[name] = content;

                    // Standard meta
                    if (name === 'description') result.description = content;
                    if (name === 'author') result.author = content;
                    if (name === 'keywords') {
                        result.keywords = content.split(',').map(k => k.trim()).filter(k => k);
                    }

                    // Open Graph
                    if (name.startsWith('og:')) {
                        const key = name.replace('og:', '');
                        result.openGraph[key] = content;
                    }

                    // Twitter Card
                    if (name.startsWith('twitter:')) {
                        const key = name.replace('twitter:', '');
                        result.twitterCard[key] = content;
                    }
                });

                // Canonical URL
                const canonical = document.querySelector('link[rel="canonical"]');
                if (canonical) {
                    result.canonical = canonical.getAttribute('href');
                }

                // Favicon
                const favicon = document.querySelector('link[rel="icon"], link[rel="shortcut icon"]');
                if (favicon) {
                    result.favicon = favicon.getAttribute('href');
                }

                // JSON-LD
                document.querySelectorAll('script[type="application/ld+json"]').forEach(script => {
                    try {
                        const data = JSON.parse(script.textContent);
                        result.jsonLd.push(data);
                    } catch (e) {}
                });

                return result;
            })()
        "#;

        let result: serde_json::Value = page
            .page
            .evaluate(script)
            .await
            .map_err(|e| ExtractionError::ExtractionFailed(e.to_string()))?
            .into_value()
            .map_err(|e| ExtractionError::ExtractionFailed(e.to_string()))?;

        let og = &result["openGraph"];
        let tw = &result["twitterCard"];

        let metadata = PageMetadata {
            title: result["title"].as_str().map(String::from),
            description: result["description"].as_str().map(String::from),
            canonical: result["canonical"].as_str().map(String::from),
            language: result["language"].as_str().map(String::from),
            author: result["author"].as_str().map(String::from),
            keywords: result["keywords"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            open_graph: OpenGraphData {
                title: og["title"].as_str().map(String::from),
                description: og["description"].as_str().map(String::from),
                image: og["image"].as_str().map(String::from),
                url: og["url"].as_str().map(String::from),
                og_type: og["type"].as_str().map(String::from),
                site_name: og["site_name"].as_str().map(String::from),
                locale: og["locale"].as_str().map(String::from),
            },
            twitter_card: TwitterCardData {
                card: tw["card"].as_str().map(String::from),
                title: tw["title"].as_str().map(String::from),
                description: tw["description"].as_str().map(String::from),
                image: tw["image"].as_str().map(String::from),
                site: tw["site"].as_str().map(String::from),
                creator: tw["creator"].as_str().map(String::from),
            },
            favicon: result["favicon"].as_str().map(String::from),
            meta_tags: result["metaTags"]
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default(),
            json_ld: result["jsonLd"].as_array().cloned().unwrap_or_default(),
        };

        debug!(
            "Extracted metadata: title={:?}, description={:?}",
            metadata.title, metadata.description
        );

        Ok(metadata)
    }

    /// Get the best title from available sources
    pub fn best_title(metadata: &PageMetadata) -> Option<String> {
        metadata
            .open_graph
            .title
            .clone()
            .or_else(|| metadata.twitter_card.title.clone())
            .or_else(|| metadata.title.clone())
    }

    /// Get the best description from available sources
    pub fn best_description(metadata: &PageMetadata) -> Option<String> {
        metadata
            .open_graph
            .description
            .clone()
            .or_else(|| metadata.twitter_card.description.clone())
            .or_else(|| metadata.description.clone())
    }

    /// Get the best image from available sources
    pub fn best_image(metadata: &PageMetadata) -> Option<String> {
        metadata
            .open_graph
            .image
            .clone()
            .or_else(|| metadata.twitter_card.image.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_metadata_default() {
        let meta = PageMetadata::default();
        assert!(meta.title.is_none());
        assert!(meta.keywords.is_empty());
        assert!(meta.json_ld.is_empty());
    }

    #[test]
    fn test_best_title() {
        let mut meta = PageMetadata {
            title: Some("Page Title".to_string()),
            ..Default::default()
        };
        meta.open_graph.title = Some("OG Title".to_string());

        // OG title should take precedence
        assert_eq!(
            MetadataExtractor::best_title(&meta),
            Some("OG Title".to_string())
        );

        // Without OG, use page title
        meta.open_graph.title = None;
        assert_eq!(
            MetadataExtractor::best_title(&meta),
            Some("Page Title".to_string())
        );
    }

    #[test]
    fn test_open_graph_data() {
        let og = OpenGraphData {
            title: Some("Title".to_string()),
            description: Some("Desc".to_string()),
            image: Some("https://example.com/img.jpg".to_string()),
            url: Some("https://example.com".to_string()),
            og_type: Some("article".to_string()),
            site_name: Some("Example".to_string()),
            locale: Some("en_US".to_string()),
        };

        assert_eq!(og.og_type, Some("article".to_string()));
    }

    #[test]
    fn test_twitter_card_data() {
        let tw = TwitterCardData {
            card: Some("summary_large_image".to_string()),
            title: Some("Title".to_string()),
            description: Some("Desc".to_string()),
            image: Some("https://example.com/img.jpg".to_string()),
            site: Some("@example".to_string()),
            creator: Some("@author".to_string()),
        };

        assert_eq!(tw.card, Some("summary_large_image".to_string()));
    }
}
