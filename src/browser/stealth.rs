//! Stealth mode for anti-detection
//!
//! This module provides techniques to make the automated browser appear
//! more like a regular user browser, bypassing common bot detection.

use crate::error::{Error, Result};
use chromiumoxide::cdp::browser_protocol::page::AddScriptToEvaluateOnNewDocumentParams;
use chromiumoxide::Page;
use tracing::{debug, instrument};

/// Stealth mode configuration and application
pub struct StealthMode;

impl StealthMode {
    /// Apply all stealth techniques to a page
    #[instrument(skip(page))]
    pub async fn apply(page: &Page) -> Result<()> {
        debug!("Applying stealth mode");

        // Apply all stealth scripts
        Self::hide_webdriver(page).await?;
        Self::mock_chrome_runtime(page).await?;
        Self::override_webgl(page).await?;
        Self::mock_plugins(page).await?;
        Self::mock_languages(page).await?;
        Self::hide_automation_indicators(page).await?;

        debug!("Stealth mode applied successfully");
        Ok(())
    }

    /// Hide navigator.webdriver property
    async fn hide_webdriver(page: &Page) -> Result<()> {
        let script = r#"
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
                configurable: true
            });
        "#;
        Self::inject_script(page, script).await
    }

    /// Mock Chrome runtime object
    async fn mock_chrome_runtime(page: &Page) -> Result<()> {
        let script = r#"
            if (!window.chrome) {
                window.chrome = {};
            }
            if (!window.chrome.runtime) {
                window.chrome.runtime = {
                    connect: function() {},
                    sendMessage: function() {},
                    onMessage: {
                        addListener: function() {},
                        removeListener: function() {}
                    }
                };
            }
        "#;
        Self::inject_script(page, script).await
    }

    /// Override WebGL fingerprinting
    async fn override_webgl(page: &Page) -> Result<()> {
        let script = r"
            const getParameterOriginal = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                // UNMASKED_VENDOR_WEBGL
                if (parameter === 37445) {
                    return 'Intel Inc.';
                }
                // UNMASKED_RENDERER_WEBGL
                if (parameter === 37446) {
                    return 'Intel Iris OpenGL Engine';
                }
                return getParameterOriginal.call(this, parameter);
            };

            // WebGL2
            if (typeof WebGL2RenderingContext !== 'undefined') {
                const getParameter2Original = WebGL2RenderingContext.prototype.getParameter;
                WebGL2RenderingContext.prototype.getParameter = function(parameter) {
                    if (parameter === 37445) {
                        return 'Intel Inc.';
                    }
                    if (parameter === 37446) {
                        return 'Intel Iris OpenGL Engine';
                    }
                    return getParameter2Original.call(this, parameter);
                };
            }
        ";
        Self::inject_script(page, script).await
    }

    /// Mock navigator.plugins
    async fn mock_plugins(page: &Page) -> Result<()> {
        let script = r#"
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    const plugins = [
                        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                        { name: 'Native Client', filename: 'internal-nacl-plugin' },
                        { name: 'Chromium PDF Plugin', filename: 'internal-pdf-viewer' },
                        { name: 'Chromium PDF Viewer', filename: 'internal-pdf-viewer' }
                    ];
                    plugins.length = 5;
                    plugins.item = (i) => plugins[i];
                    plugins.namedItem = (name) => plugins.find(p => p.name === name);
                    plugins.refresh = () => {};
                    return plugins;
                },
                configurable: true
            });
        "#;
        Self::inject_script(page, script).await
    }

    /// Mock navigator.languages
    async fn mock_languages(page: &Page) -> Result<()> {
        let script = r#"
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en', 'es'],
                configurable: true
            });

            Object.defineProperty(navigator, 'language', {
                get: () => 'en-US',
                configurable: true
            });
        "#;
        Self::inject_script(page, script).await
    }

    /// Hide other automation indicators
    async fn hide_automation_indicators(page: &Page) -> Result<()> {
        let script = r#"
            // Hide automation flags
            Object.defineProperty(navigator, 'maxTouchPoints', {
                get: () => 0,
                configurable: true
            });

            // Override permissions API
            if (navigator.permissions) {
                const originalQuery = navigator.permissions.query;
                navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            }

            // Mock connection type
            if (!navigator.connection) {
                Object.defineProperty(navigator, 'connection', {
                    get: () => ({
                        effectiveType: '4g',
                        rtt: 50,
                        downlink: 10,
                        saveData: false
                    }),
                    configurable: true
                });
            }

            // Hide headless indicators in User-Agent Client Hints
            if (navigator.userAgentData) {
                Object.defineProperty(navigator.userAgentData, 'brands', {
                    get: () => [
                        { brand: 'Google Chrome', version: '120' },
                        { brand: 'Chromium', version: '120' },
                        { brand: 'Not_A Brand', version: '24' }
                    ],
                    configurable: true
                });
            }
        "#;
        Self::inject_script(page, script).await
    }

    /// Inject a script to run on new document
    async fn inject_script(page: &Page, script: &str) -> Result<()> {
        let params = AddScriptToEvaluateOnNewDocumentParams::builder()
            .source(script)
            .build()
            .map_err(|e| Error::cdp(format!("Failed to build script params: {}", e)))?;

        page.execute(params)
            .await
            .map_err(|e| Error::cdp(format!("Failed to inject script: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Stealth mode tests require a running browser
    // These are integration tests that will be in tests/browser_tests.rs
}
