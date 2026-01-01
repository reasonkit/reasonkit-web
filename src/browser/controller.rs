//! Browser lifecycle management
//!
//! This module handles browser launch, shutdown, and page management.

use crate::error::{BrowserError, Error, Result};
use chromiumoxide::browser::{Browser, BrowserConfig as CdpBrowserConfig};
use chromiumoxide::Page;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument, warn};

/// Configuration for browser launch
#[derive(Debug, Clone)]
pub struct BrowserConfig {
    /// Run in headless mode (default: true)
    pub headless: bool,
    /// Browser window width (default: 1920)
    pub width: u32,
    /// Browser window height (default: 1080)
    pub height: u32,
    /// Enable sandbox (default: true for production)
    pub sandbox: bool,
    /// User agent string (None = use default)
    pub user_agent: Option<String>,
    /// Navigation timeout in milliseconds (default: 30000)
    pub timeout_ms: u64,
    /// Path to Chrome/Chromium executable (None = auto-detect)
    pub chrome_path: Option<String>,
    /// Enable stealth mode (default: true)
    pub stealth: bool,
    /// Additional Chrome arguments
    pub extra_args: Vec<String>,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            headless: true,
            width: 1920,
            height: 1080,
            sandbox: true,
            user_agent: None,
            timeout_ms: 30000,
            chrome_path: None,
            stealth: true,
            extra_args: Vec::new(),
        }
    }
}

impl BrowserConfig {
    /// Create a new config builder
    pub fn builder() -> BrowserConfigBuilder {
        BrowserConfigBuilder::default()
    }
}

/// Builder for BrowserConfig
#[derive(Default)]
pub struct BrowserConfigBuilder {
    config: BrowserConfig,
}

impl BrowserConfigBuilder {
    /// Set headless mode
    pub fn headless(mut self, headless: bool) -> Self {
        self.config.headless = headless;
        self
    }

    /// Set viewport dimensions
    pub fn viewport(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    /// Enable/disable sandbox
    pub fn sandbox(mut self, sandbox: bool) -> Self {
        self.config.sandbox = sandbox;
        self
    }

    /// Set user agent
    pub fn user_agent<S: Into<String>>(mut self, ua: S) -> Self {
        self.config.user_agent = Some(ua.into());
        self
    }

    /// Set navigation timeout
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.config.timeout_ms = ms;
        self
    }

    /// Set Chrome path
    pub fn chrome_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.chrome_path = Some(path.into());
        self
    }

    /// Enable/disable stealth mode
    pub fn stealth(mut self, stealth: bool) -> Self {
        self.config.stealth = stealth;
        self
    }

    /// Add extra Chrome argument
    pub fn arg<S: Into<String>>(mut self, arg: S) -> Self {
        self.config.extra_args.push(arg.into());
        self
    }

    /// Build the config
    pub fn build(self) -> BrowserConfig {
        self.config
    }
}

/// Handle to an open browser page
#[derive(Clone)]
pub struct PageHandle {
    pub(crate) page: Page,
    pub(crate) url: Arc<RwLock<String>>,
}

impl PageHandle {
    /// Get the underlying chromiumoxide Page
    pub fn inner(&self) -> &Page {
        &self.page
    }

    /// Get the current URL
    pub async fn url(&self) -> String {
        self.url.read().await.clone()
    }

    /// Set the current URL (internal use)
    pub(crate) async fn set_url(&self, url: String) {
        *self.url.write().await = url;
    }
}

/// High-level browser controller
pub struct BrowserController {
    browser: Browser,
    handler: JoinHandle<()>,
    config: BrowserConfig,
    pages: Arc<RwLock<Vec<PageHandle>>>,
}

impl BrowserController {
    /// Create a new browser controller with default config
    #[instrument]
    pub async fn new() -> Result<Self> {
        Self::with_config(BrowserConfig::default()).await
    }

    /// Create a new browser controller with custom config
    #[instrument(skip(config))]
    pub async fn with_config(config: BrowserConfig) -> Result<Self> {
        info!(
            "Launching browser with config: headless={}",
            config.headless
        );

        let mut builder = CdpBrowserConfig::builder();

        // Set viewport
        builder = builder.viewport(chromiumoxide::handler::viewport::Viewport {
            width: config.width,
            height: config.height,
            device_scale_factor: None,
            emulating_mobile: false,
            is_landscape: true,
            has_touch: false,
        });

        // Headless mode
        if config.headless {
            builder = builder.with_head();
        }

        // Sandbox
        if !config.sandbox {
            builder = builder.arg("--no-sandbox");
        }

        // Chrome path
        if let Some(ref path) = config.chrome_path {
            builder = builder.chrome_executable(path);
        }

        // Extra args
        for arg in &config.extra_args {
            builder = builder.arg(arg);
        }

        // Ensure WebGL support in headless mode
        if config.headless {
            builder = builder.arg("--use-gl=swiftshader");
            builder = builder.arg("--enable-webgl");
            builder = builder.arg("--ignore-gpu-blocklist");
        }

        let cdp_config = builder
            .build()
            .map_err(|e| BrowserError::ConfigError(e.to_string()))?;

        let (browser, mut handler) = Browser::launch(cdp_config)
            .await
            .map_err(|e| BrowserError::LaunchFailed(e.to_string()))?;

        // Spawn handler task
        let handler_task = tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                if event.is_err() {
                    warn!("Browser handler event error");
                    break;
                }
            }
            debug!("Browser handler finished");
        });

        info!("Browser launched successfully");

        Ok(Self {
            browser,
            handler: handler_task,
            config,
            pages: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Create a new page/tab
    #[instrument(skip(self))]
    pub async fn new_page(&self) -> Result<PageHandle> {
        let page = self
            .browser
            .new_page("about:blank")
            .await
            .map_err(|e| BrowserError::PageCreationFailed(e.to_string()))?;

        // Apply stealth mode if enabled
        if self.config.stealth {
            super::stealth::StealthMode::apply(&page).await?;
        }

        let handle = PageHandle {
            page,
            url: Arc::new(RwLock::new("about:blank".to_string())),
        };

        self.pages.write().await.push(handle.clone());
        debug!("Created new page");

        Ok(handle)
    }

    /// Navigate to URL and return page handle
    #[instrument(skip(self))]
    pub async fn navigate(&self, url: &str) -> Result<PageHandle> {
        let page_handle = self.new_page().await?;
        super::navigation::PageNavigator::goto(&page_handle, url, None).await?;
        Ok(page_handle)
    }

    /// Get the browser configuration
    pub fn config(&self) -> &BrowserConfig {
        &self.config
    }

    /// Get the number of open pages
    pub async fn page_count(&self) -> usize {
        self.pages.read().await.len()
    }

    /// Close the browser
    #[instrument(skip(self))]
    pub async fn close(mut self) -> Result<()> {
        info!("Closing browser");

        // Clear pages (browser close will close all pages)
        self.pages.write().await.clear();

        // Close browser
        self.browser
            .close()
            .await
            .map_err(|e| Error::cdp(e.to_string()))?;

        // Wait for handler to finish
        let _ = tokio::time::timeout(Duration::from_secs(5), self.handler).await;

        info!("Browser closed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_config_default() {
        let config = BrowserConfig::default();
        assert!(config.headless);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert!(config.sandbox);
        assert!(config.stealth);
        assert_eq!(config.timeout_ms, 30000);
    }

    #[test]
    fn test_browser_config_builder() {
        let config = BrowserConfig::builder()
            .headless(false)
            .viewport(1280, 720)
            .sandbox(false)
            .user_agent("TestBot/1.0")
            .timeout_ms(60000)
            .stealth(false)
            .arg("--disable-gpu")
            .build();

        assert!(!config.headless);
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert!(!config.sandbox);
        assert_eq!(config.user_agent, Some("TestBot/1.0".to_string()));
        assert_eq!(config.timeout_ms, 60000);
        assert!(!config.stealth);
        assert_eq!(config.extra_args, vec!["--disable-gpu"]);
    }
}
