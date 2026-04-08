use crate::api::{ApiClient, QueryRequest};
use crate::config::Settings;
use crate::error::Result;
use crate::output::{format_query_response, OutputFormat};

#[derive(Debug)]
pub struct QueryCommand {
    pub query: String,
    pub top_k: Option<i32>,
    pub format: OutputFormat,
    pub base_url: Option<String>,
}

impl QueryCommand {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            top_k: None,
            format: OutputFormat::default(),
            base_url: None,
        }
    }

    pub fn with_top_k(mut self, k: i32) -> Self {
        self.top_k = Some(k.clamp(1, 50));
        self
    }

    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub async fn execute(&self, settings: &Settings) -> Result<()> {
        // Create API client
        let client = if let Some(ref url) = self.base_url {
            ApiClient::new(settings)?.with_base_url(url)
        } else {
            ApiClient::new(settings)?
        };

        // Check connection
        if !client.is_connected().await {
            crate::output::print_error(&format!(
                "Cannot connect to API at {}",
                settings.api.base_url
            ));
            crate::output::print_error("Make sure the FastAPI server is running.");
            return Ok(());
        }

        // Build request
        let mut request = QueryRequest::new(&self.query);
        if let Some(k) = self.top_k {
            request = request.with_top_k(k);
        }

        // Execute query
        let response = client.query(request).await?;

        // Format and print output
        let output = format_query_response(&response, self.format);
        print!("{}", output);

        Ok(())
    }
}