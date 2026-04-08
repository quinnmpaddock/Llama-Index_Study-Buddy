use colored::Colorize;
use tabled::{settings::Style, Table, Tabled};

use crate::api::{ApiClient, SummaryListResponse, SummaryVersion, SummaryVersionInfo};
use crate::config::Settings;
use crate::error::Result;
use crate::output::{print_error, print_separator, print_status, OutputFormat};

/// Summaries command for managing community summaries
pub struct SummariesCommand {
    format: OutputFormat,
}

impl SummariesCommand {
    pub fn new() -> Self {
        Self {
            format: OutputFormat::Table,
        }
    }

    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    /// List all summary versions
    pub async fn list(&self, settings: &Settings) -> Result<()> {
        let client = ApiClient::new(settings)?;

        print_status("Fetching", "summary versions...");

        let response = client.list_summaries().await?;

        self.display_list(&response);

        Ok(())
    }

    /// Show the current summary version
    pub async fn current(&self, settings: &Settings) -> Result<()> {
        let client = ApiClient::new(settings)?;

        print_status("Fetching", "current summary version...");

        match client.get_current_summary().await {
            Ok(version) => {
                self.display_version(&version);
            }
            Err(e) => {
                print_error(&format!("No current summary version found: {}", e));
            }
        }

        Ok(())
    }

    /// Delete old summary versions, keeping N most recent
    pub async fn cleanup(&self, settings: &Settings, keep: i32) -> Result<()> {
        let client = ApiClient::new(settings)?;

        print_status("Cleaning up", &format!("summary versions (keeping {} most recent)...", keep));

        let response = client.cleanup_summaries(keep).await?;

        self.display_cleanup(&response);

        Ok(())
    }

    /// Display the list of summary versions
    fn display_list(&self, response: &SummaryListResponse) {
        match self.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(response).unwrap_or_default());
            }
            OutputFormat::Yaml => {
                if let Some(ref current) = response.current {
                    println!("current:");
                    println!("  version: {}", current.version);
                    println!("  created_at: {}", current.created_at);
                }
                println!("versions:");
                for v in &response.versions {
                    println!("  - version: {}", v.version);
                    println!("    modified: {}", v.modified);
                    println!("    size: {} bytes", v.size_bytes);
                }
            }
            OutputFormat::Table => {
                print_separator();
                
                // Show current version
                if let Some(ref current) = response.current {
                    println!("{}", "Current Version:".green().bold());
                    println!("  {}: {}", "Version".cyan(), current.version);
                    println!("  {}: {}", "Created".cyan(), current.created_at);
                    if let Some(stats) = current.stats.get("total_entities") {
                        println!("  {}: {}", "Entities".cyan(), stats);
                    }
                    if let Some(stats) = current.stats.get("total_communities") {
                        println!("  {}: {}", "Communities".cyan(), stats);
                    }
                    print_separator();
                }

                // Show all versions
                if response.versions.is_empty() {
                    println!("{}", "No summary versions found.".yellow());
                } else {
                    println!("{}", "All Versions:".green().bold());
                    display_versions_table(&response.versions);
                }
            }
        }
    }

    /// Display a single version's details
    fn display_version(&self, version: &SummaryVersion) {
        match self.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(version).unwrap_or_default());
            }
            OutputFormat::Yaml => {
                println!("version: {}", version.version);
                println!("created_at: {}", version.created_at);
                println!("files:");
                for (k, v) in &version.files {
                    println!("  {}: {}", k, v);
                }
                println!("stats:");
                for (k, v) in &version.stats {
                    println!("  {}: {}", k, v);
                }
            }
            OutputFormat::Table => {
                print_separator();
                println!("{}", "Current Summary Version".green().bold());
                println!("{}: {}", "Version".cyan().bold(), version.version);
                println!("{}: {}", "Created".cyan().bold(), version.created_at);
                println!();
                println!("{}", "Files:".cyan().bold());
                for (name, file) in &version.files {
                    println!("  {}: {}", name, file);
                }
                println!();
                println!("{}", "Stats:".cyan().bold());
                for (name, value) in &version.stats {
                    println!("  {}: {}", name, value);
                }
            }
        }
    }

    /// Display cleanup results
    fn display_cleanup(&self, response: &crate::api::SummaryCleanupResponse) {
        match self.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(response).unwrap_or_default());
            }
            OutputFormat::Yaml => {
                println!("message: {}", response.message);
                println!("deleted:");
                for f in &response.deleted {
                    println!("  - {}", f);
                }
                println!("kept:");
                for f in &response.kept {
                    println!("  - {}", f);
                }
            }
            OutputFormat::Table => {
                print_separator();
                println!("{}", "Cleanup Complete".green().bold());
                println!("{}: {}", "Message".cyan().bold(), response.message);
                
                if !response.deleted.is_empty() {
                    println!();
                    println!("{}", "Deleted:".red().bold());
                    for f in &response.deleted {
                        println!("  {} {}", "✗".red(), f);
                    }
                }
                
                if !response.kept.is_empty() {
                    println!();
                    println!("{}", "Kept:".green().bold());
                    for f in &response.kept {
                        println!("  {} {}", "✓".green(), f);
                    }
                }
            }
        }
    }
}

/// Display versions as a table
fn display_versions_table(versions: &[SummaryVersionInfo]) {
    #[derive(Tabled)]
    struct VersionRow {
        #[tabled(rename = "#")]
        id: usize,
        #[tabled(rename = "Version")]
        version: String,
        #[tabled(rename = "Modified")]
        modified: String,
        #[tabled(rename = "Size")]
        size: String,
    }

    let rows: Vec<VersionRow> = versions
        .iter()
        .enumerate()
        .map(|(i, v)| VersionRow {
            id: i + 1,
            version: v.version.clone(),
            modified: v.modified.split('T').next().unwrap_or(&v.modified).to_string(),
            size: format_size(v.size_bytes),
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
}

/// Format file size in human-readable format
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;

    if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}