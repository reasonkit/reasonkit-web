//! Page capture functionality
//!
//! This module handles screenshots, PDFs, MHTML snapshots, and HTML capture.

use crate::browser::PageHandle;
use crate::error::{CaptureError, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chromiumoxide::cdp::browser_protocol::page::{
    CaptureScreenshotFormat, CaptureSnapshotFormat, CaptureSnapshotParams, PrintToPdfParams,
};
use chromiumoxide::page::ScreenshotParams;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

/// Format for captures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CaptureFormat {
    /// PNG screenshot
    #[default]
    Png,
    /// JPEG screenshot
    Jpeg,
    /// WebP screenshot
    Webp,
    /// PDF document
    Pdf,
    /// MHTML archive
    Mhtml,
    /// Raw HTML
    Html,
}

/// Options for capture operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureOptions {
    /// Capture format
    #[serde(default)]
    pub format: CaptureFormat,
    /// JPEG/WebP quality (0-100)
    #[serde(default = "default_quality")]
    pub quality: u8,
    /// Capture full page (not just viewport)
    #[serde(default = "default_true")]
    pub full_page: bool,
    /// Viewport width for capture
    pub width: Option<u32>,
    /// Viewport height for capture
    pub height: Option<u32>,
    /// CSS selector to clip to
    pub clip_selector: Option<String>,
    /// Return as base64 instead of bytes
    #[serde(default)]
    pub as_base64: bool,
}

fn default_quality() -> u8 {
    85
}

fn default_true() -> bool {
    true
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            format: CaptureFormat::Png,
            quality: 85,
            full_page: true,
            width: None,
            height: None,
            clip_selector: None,
            as_base64: false,
        }
    }
}

impl CaptureOptions {
    /// Create options for PNG screenshot
    pub fn png() -> Self {
        Self {
            format: CaptureFormat::Png,
            ..Default::default()
        }
    }

    /// Create options for JPEG screenshot
    pub fn jpeg(quality: u8) -> Self {
        Self {
            format: CaptureFormat::Jpeg,
            quality,
            ..Default::default()
        }
    }

    /// Create options for PDF
    pub fn pdf() -> Self {
        Self {
            format: CaptureFormat::Pdf,
            ..Default::default()
        }
    }

    /// Create options for MHTML
    pub fn mhtml() -> Self {
        Self {
            format: CaptureFormat::Mhtml,
            ..Default::default()
        }
    }

    /// Create options for HTML
    pub fn html() -> Self {
        Self {
            format: CaptureFormat::Html,
            ..Default::default()
        }
    }

    /// Validate capture request options
    pub fn validate(&self) -> std::result::Result<(), String> {
        // Quality must be in valid range
        if self.quality > 100 {
            return Err("Quality must be between 0 and 100".to_string());
        }

        // Width/height constraints
        if let Some(w) = self.width {
            if w == 0 || w > 16384 {
                return Err("Width must be between 1 and 16384".to_string());
            }
        }
        if let Some(h) = self.height {
            if h == 0 || h > 16384 {
                return Err("Height must be between 1 and 16384".to_string());
            }
        }

        Ok(())
    }
}

/// Result of a capture operation
#[derive(Debug, Clone)]
pub struct CaptureResult {
    /// The captured data
    pub data: Vec<u8>,
    /// The format of the capture
    pub format: CaptureFormat,
    /// Base64 encoded data (if requested)
    pub base64: Option<String>,
    /// Width of the capture (for images)
    pub width: Option<u32>,
    /// Height of the capture (for images)
    pub height: Option<u32>,
    /// Size in bytes
    pub size: usize,
}

impl CaptureResult {
    /// Get data as base64
    pub fn to_base64(&self) -> String {
        BASE64.encode(&self.data)
    }

    /// Get appropriate MIME type
    pub fn mime_type(&self) -> &'static str {
        match self.format {
            CaptureFormat::Png => "image/png",
            CaptureFormat::Jpeg => "image/jpeg",
            CaptureFormat::Webp => "image/webp",
            CaptureFormat::Pdf => "application/pdf",
            CaptureFormat::Mhtml => "multipart/related",
            CaptureFormat::Html => "text/html",
        }
    }

    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self.format {
            CaptureFormat::Png => "png",
            CaptureFormat::Jpeg => "jpg",
            CaptureFormat::Webp => "webp",
            CaptureFormat::Pdf => "pdf",
            CaptureFormat::Mhtml => "mhtml",
            CaptureFormat::Html => "html",
        }
    }
}

/// Page capture functionality
pub struct PageCapture;

impl PageCapture {
    /// Capture a page with the given options
    #[instrument(skip(page))]
    pub async fn capture(page: &PageHandle, options: &CaptureOptions) -> Result<CaptureResult> {
        match options.format {
            CaptureFormat::Png | CaptureFormat::Jpeg | CaptureFormat::Webp => {
                Self::screenshot(page, options).await
            }
            CaptureFormat::Pdf => Self::pdf(page, options).await,
            CaptureFormat::Mhtml => Self::mhtml(page).await,
            CaptureFormat::Html => Self::html(page).await,
        }
    }

    /// Take a screenshot
    #[instrument(skip(page))]
    pub async fn screenshot(page: &PageHandle, options: &CaptureOptions) -> Result<CaptureResult> {
        info!("Capturing screenshot");

        let format = match options.format {
            CaptureFormat::Png => CaptureScreenshotFormat::Png,
            CaptureFormat::Jpeg => CaptureScreenshotFormat::Jpeg,
            CaptureFormat::Webp => CaptureScreenshotFormat::Webp,
            _ => CaptureScreenshotFormat::Png,
        };

        let mut params_builder = ScreenshotParams::builder()
            .format(format)
            .from_surface(true)
            .capture_beyond_viewport(options.full_page);

        // Set quality for JPEG/WebP
        if matches!(options.format, CaptureFormat::Jpeg | CaptureFormat::Webp) {
            params_builder = params_builder.quality(options.quality as i64);
        }

        let params = params_builder.build();

        let data = page
            .page
            .screenshot(params)
            .await
            .map_err(|e| CaptureError::ScreenshotFailed(e.to_string()))?;

        let size = data.len();
        debug!("Screenshot captured: {} bytes", size);

        let base64 = if options.as_base64 {
            Some(BASE64.encode(&data))
        } else {
            None
        };

        Ok(CaptureResult {
            data,
            format: options.format,
            base64,
            width: options.width,
            height: options.height,
            size,
        })
    }

    /// Generate a PDF
    #[instrument(skip(page))]
    pub async fn pdf(page: &PageHandle, options: &CaptureOptions) -> Result<CaptureResult> {
        info!("Generating PDF");

        let mut params_builder = PrintToPdfParams::builder()
            .print_background(true)
            .prefer_css_page_size(true);

        // Set page size if specified
        if let (Some(width), Some(height)) = (options.width, options.height) {
            params_builder = params_builder
                .paper_width(width as f64 / 96.0) // Convert pixels to inches
                .paper_height(height as f64 / 96.0);
        }

        let params = params_builder.build();

        let data = page
            .page
            .pdf(params)
            .await
            .map_err(|e| CaptureError::PdfFailed(e.to_string()))?;

        let size = data.len();
        debug!("PDF generated: {} bytes", size);

        let base64 = if options.as_base64 {
            Some(BASE64.encode(&data))
        } else {
            None
        };

        Ok(CaptureResult {
            data,
            format: CaptureFormat::Pdf,
            base64,
            width: options.width,
            height: options.height,
            size,
        })
    }

    /// Capture MHTML archive
    #[instrument(skip(page))]
    pub async fn mhtml(page: &PageHandle) -> Result<CaptureResult> {
        info!("Capturing MHTML");

        let params = CaptureSnapshotParams::builder()
            .format(CaptureSnapshotFormat::Mhtml)
            .build();

        let result = page
            .page
            .execute(params)
            .await
            .map_err(|e| CaptureError::MhtmlFailed(e.to_string()))?;

        let data = result.data.clone().into_bytes();
        let size = data.len();
        debug!("MHTML captured: {} bytes", size);

        Ok(CaptureResult {
            data,
            format: CaptureFormat::Mhtml,
            base64: None,
            width: None,
            height: None,
            size,
        })
    }

    /// Capture raw HTML
    #[instrument(skip(page))]
    pub async fn html(page: &PageHandle) -> Result<CaptureResult> {
        info!("Capturing HTML");

        let html: String = page
            .page
            .evaluate("document.documentElement.outerHTML")
            .await
            .map_err(|e| CaptureError::HtmlFailed(e.to_string()))?
            .into_value()
            .map_err(|e| CaptureError::HtmlFailed(e.to_string()))?;

        let data = html.into_bytes();
        let size = data.len();
        debug!("HTML captured: {} bytes", size);

        Ok(CaptureResult {
            data,
            format: CaptureFormat::Html,
            base64: None,
            width: None,
            height: None,
            size,
        })
    }

    /// Capture a specific element
    #[instrument(skip(page))]
    pub async fn element_screenshot(
        page: &PageHandle,
        selector: &str,
        format: CaptureFormat,
    ) -> Result<CaptureResult> {
        info!("Capturing element: {}", selector);

        let element = page
            .page
            .find_element(selector)
            .await
            .map_err(|e| CaptureError::ScreenshotFailed(format!("Element not found: {}", e)))?;

        let cdp_format = match format {
            CaptureFormat::Png => CaptureScreenshotFormat::Png,
            CaptureFormat::Jpeg => CaptureScreenshotFormat::Jpeg,
            CaptureFormat::Webp => CaptureScreenshotFormat::Webp,
            _ => CaptureScreenshotFormat::Png,
        };

        let data = element
            .screenshot(cdp_format)
            .await
            .map_err(|e| CaptureError::ScreenshotFailed(e.to_string()))?;

        let size = data.len();
        debug!("Element screenshot captured: {} bytes", size);

        Ok(CaptureResult {
            data,
            format,
            base64: None,
            width: None,
            height: None,
            size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // CaptureOptions Tests
    // ========================================================================

    #[test]
    fn test_capture_options_default() {
        let opts = CaptureOptions::default();
        assert_eq!(opts.format, CaptureFormat::Png);
        assert_eq!(opts.quality, 85);
        assert!(opts.full_page);
        assert!(!opts.as_base64);
        assert!(opts.width.is_none());
        assert!(opts.height.is_none());
        assert!(opts.clip_selector.is_none());
    }

    #[test]
    fn test_capture_format_factories() {
        let png = CaptureOptions::png();
        assert_eq!(png.format, CaptureFormat::Png);

        let jpeg = CaptureOptions::jpeg(90);
        assert_eq!(jpeg.format, CaptureFormat::Jpeg);
        assert_eq!(jpeg.quality, 90);

        let pdf = CaptureOptions::pdf();
        assert_eq!(pdf.format, CaptureFormat::Pdf);

        let mhtml = CaptureOptions::mhtml();
        assert_eq!(mhtml.format, CaptureFormat::Mhtml);

        let html = CaptureOptions::html();
        assert_eq!(html.format, CaptureFormat::Html);
    }

    #[test]
    fn test_validate_capture_request_valid() {
        let opts = CaptureOptions {
            format: CaptureFormat::Png,
            quality: 85,
            full_page: true,
            width: Some(1920),
            height: Some(1080),
            clip_selector: None,
            as_base64: false,
        };
        assert!(opts.validate().is_ok());
    }

    #[test]
    fn test_validate_capture_request_valid_minimal() {
        let opts = CaptureOptions::default();
        assert!(opts.validate().is_ok());
    }

    #[test]
    fn test_validate_capture_request_quality_too_high() {
        let mut opts = CaptureOptions::default();
        opts.quality = 101;
        let result = opts.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Quality"));
    }

    #[test]
    fn test_validate_capture_request_width_too_large() {
        let mut opts = CaptureOptions::default();
        opts.width = Some(20000);
        let result = opts.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Width"));
    }

    #[test]
    fn test_validate_capture_request_height_zero() {
        let mut opts = CaptureOptions::default();
        opts.height = Some(0);
        let result = opts.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Height"));
    }

    #[test]
    fn test_validate_capture_request_max_dimensions() {
        let mut opts = CaptureOptions::default();
        opts.width = Some(16384);
        opts.height = Some(16384);
        assert!(opts.validate().is_ok());
    }

    // ========================================================================
    // CaptureResult Tests
    // ========================================================================

    #[test]
    fn test_capture_result_mime_type() {
        let formats_and_mimes = [
            (CaptureFormat::Png, "image/png"),
            (CaptureFormat::Jpeg, "image/jpeg"),
            (CaptureFormat::Webp, "image/webp"),
            (CaptureFormat::Pdf, "application/pdf"),
            (CaptureFormat::Mhtml, "multipart/related"),
            (CaptureFormat::Html, "text/html"),
        ];

        for (format, expected_mime) in formats_and_mimes {
            let result = CaptureResult {
                data: vec![],
                format,
                base64: None,
                width: None,
                height: None,
                size: 0,
            };
            assert_eq!(result.mime_type(), expected_mime);
        }
    }

    #[test]
    fn test_capture_result_extension() {
        let formats_and_exts = [
            (CaptureFormat::Png, "png"),
            (CaptureFormat::Jpeg, "jpg"),
            (CaptureFormat::Webp, "webp"),
            (CaptureFormat::Pdf, "pdf"),
            (CaptureFormat::Mhtml, "mhtml"),
            (CaptureFormat::Html, "html"),
        ];

        for (format, expected_ext) in formats_and_exts {
            let result = CaptureResult {
                data: vec![],
                format,
                base64: None,
                width: None,
                height: None,
                size: 0,
            };
            assert_eq!(result.extension(), expected_ext);
        }
    }

    #[test]
    fn test_capture_result_base64() {
        let result = CaptureResult {
            data: b"hello".to_vec(),
            format: CaptureFormat::Png,
            base64: None,
            width: None,
            height: None,
            size: 5,
        };
        assert_eq!(result.to_base64(), "aGVsbG8=");
    }

    #[test]
    fn test_capture_result_base64_empty() {
        let result = CaptureResult {
            data: vec![],
            format: CaptureFormat::Png,
            base64: None,
            width: None,
            height: None,
            size: 0,
        };
        assert_eq!(result.to_base64(), "");
    }

    #[test]
    fn test_capture_result_base64_binary() {
        let result = CaptureResult {
            data: vec![0x89, 0x50, 0x4E, 0x47], // PNG header
            format: CaptureFormat::Png,
            base64: None,
            width: None,
            height: None,
            size: 4,
        };
        let b64 = result.to_base64();
        assert!(!b64.is_empty());
        // Verify it decodes back correctly
        let decoded = BASE64.decode(&b64).unwrap();
        assert_eq!(decoded, vec![0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn test_capture_result_with_dimensions() {
        let result = CaptureResult {
            data: vec![1, 2, 3],
            format: CaptureFormat::Png,
            base64: None,
            width: Some(1920),
            height: Some(1080),
            size: 3,
        };
        assert_eq!(result.width, Some(1920));
        assert_eq!(result.height, Some(1080));
        assert_eq!(result.size, 3);
    }

    #[test]
    fn test_capture_result_with_precomputed_base64() {
        let result = CaptureResult {
            data: b"hello".to_vec(),
            format: CaptureFormat::Png,
            base64: Some("precomputed".to_string()),
            width: None,
            height: None,
            size: 5,
        };
        // base64 field is stored separately from to_base64() computation
        assert_eq!(result.base64, Some("precomputed".to_string()));
        // to_base64() still computes from data
        assert_eq!(result.to_base64(), "aGVsbG8=");
    }

    // ========================================================================
    // CaptureFormat Tests
    // ========================================================================

    #[test]
    fn test_capture_format_default() {
        let format = CaptureFormat::default();
        assert_eq!(format, CaptureFormat::Png);
    }

    #[test]
    fn test_capture_format_serialization() {
        let format = CaptureFormat::Jpeg;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"jpeg\"");

        let deserialized: CaptureFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, CaptureFormat::Jpeg);
    }

    #[test]
    fn test_capture_format_all_variants_serialize() {
        let formats = [
            (CaptureFormat::Png, "\"png\""),
            (CaptureFormat::Jpeg, "\"jpeg\""),
            (CaptureFormat::Webp, "\"webp\""),
            (CaptureFormat::Pdf, "\"pdf\""),
            (CaptureFormat::Mhtml, "\"mhtml\""),
            (CaptureFormat::Html, "\"html\""),
        ];

        for (format, expected_json) in formats {
            let json = serde_json::to_string(&format).unwrap();
            assert_eq!(json, expected_json);
        }
    }

    #[test]
    fn test_capture_format_equality() {
        assert_eq!(CaptureFormat::Png, CaptureFormat::Png);
        assert_ne!(CaptureFormat::Png, CaptureFormat::Jpeg);
    }

    #[test]
    fn test_capture_format_clone() {
        let format = CaptureFormat::Webp;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    // ========================================================================
    // CaptureOptions Serialization Tests
    // ========================================================================

    #[test]
    fn test_capture_options_serialization() {
        let opts = CaptureOptions {
            format: CaptureFormat::Jpeg,
            quality: 90,
            full_page: false,
            width: Some(800),
            height: Some(600),
            clip_selector: Some("#main".to_string()),
            as_base64: true,
        };

        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"jpeg\""));
        assert!(json.contains("90"));
        assert!(json.contains("#main"));

        let deserialized: CaptureOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.format, CaptureFormat::Jpeg);
        assert_eq!(deserialized.quality, 90);
        assert!(!deserialized.full_page);
        assert!(deserialized.as_base64);
    }

    #[test]
    fn test_capture_options_deserialize_with_defaults() {
        let json = r#"{"format": "png"}"#;
        let opts: CaptureOptions = serde_json::from_str(json).unwrap();

        // Check defaults are applied
        assert_eq!(opts.format, CaptureFormat::Png);
        assert_eq!(opts.quality, 85); // default
        assert!(opts.full_page); // default true
        assert!(!opts.as_base64); // default false
    }

    #[test]
    fn test_capture_options_jpeg_quality_boundary() {
        let opts_min = CaptureOptions::jpeg(0);
        assert_eq!(opts_min.quality, 0);
        assert!(opts_min.validate().is_ok());

        let opts_max = CaptureOptions::jpeg(100);
        assert_eq!(opts_max.quality, 100);
        assert!(opts_max.validate().is_ok());
    }
}
