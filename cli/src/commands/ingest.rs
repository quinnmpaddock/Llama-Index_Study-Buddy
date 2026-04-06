use std::io::{self, BufRead, Write};
use std::time::Duration;

use colored::Colorize;
use tabled::{settings::Style, Table, Tabled};

use crate::api::{ApiClient, IngestFileInfo, IngestRequest, IngestResponse, IngestStatus};
use crate::config::Settings;
use crate::error::Result;
use crate::output::{print_error, print_separator, print_status, OutputFormat};

/// Ingest command for document ingestion
pub struct IngestCommand {
    directory: String,
    files: Option<Vec<String>>,
    format: OutputFormat,
    yes: bool, // Skip confirmation
}

impl IngestCommand {
    pub fn new(directory: impl Into<String>) -> Self {
        Self {
            directory: directory.into(),
            files: None,
            format: OutputFormat::Table,
            yes: false,
        }
    }

    pub fn with_files(mut self, files: Vec<String>) -> Self {
        self.files = Some(files);
        self
    }

    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_yes(mut self, yes: bool) -> Self {
        self.yes = yes;
        self
    }

    pub async fn execute(&self, settings: &Settings) -> Result<()> {
        let client = ApiClient::new(settings)?;

        // Resolve to absolute path
        let abs_path = std::fs::canonicalize(&self.directory)
            .unwrap_or_else(|_| std::path::PathBuf::from(&self.directory));

        print_status("Scanning", &format!("{}...", abs_path.display()));
        print_separator();

        // Get preview from API
        let preview = client.preview_ingest(abs_path.to_str().unwrap()).await?;

        // Display directory info
        println!("{} {}", "Directory:".cyan().bold(), preview.directory);
        println!(
            "{} {}",
            "Supported:".cyan().bold(),
            preview.supported_extensions.join(", ")
        );
        print_separator();

        if preview.files.is_empty() {
            print_error("No supported files found in directory.");
            return Ok(());
        }

        // Filter files if specific ones were requested
        let files_to_process = if let Some(ref requested) = self.files {
            let filtered: Vec<&IngestFileInfo> = preview
                .files
                .iter()
                .filter(|f| requested.contains(&f.name))
                .collect();

            if filtered.is_empty() {
                print_error(&format!(
                    "None of the requested files found: {}",
                    requested.join(", ")
                ));
                return Ok(());
            }

            if filtered.len() != requested.len() {
                let found: Vec<&str> = filtered.iter().map(|f| f.name.as_str()).collect();
                let missing: Vec<&str> = requested
                    .iter()
                    .filter(|r| !found.contains(&r.as_str()))
                    .map(|s| s.as_str())
                    .collect();
                println!(
                    "{} {}",
                    "Warning:".yellow().bold(),
                    format!("Files not found: {}", missing.join(", "))
                );
            }

            filtered.into_iter().cloned().collect()
        } else {
            preview.files.clone()
        };

        // Display files to be processed
        println!("{}", "Files to ingest:".green().bold());
        display_files_table(&files_to_process);
        print_separator();

        println!(
            "{} {}",
            "Total:".cyan().bold(),
            format!("{} file(s), {:.2} KB", files_to_process.len(), total_size_kb(&files_to_process))
        );

        // Get confirmation (unless --yes flag)
        if !self.yes {
            let prompt = if self.files.is_some() {
                "Proceed with ingestion? [y/N]"
            } else {
                "Ingest ALL files in directory? [y/N]"
            };

            if !confirm(prompt)? {
                println!("{}", "Cancelled.".yellow());
                return Ok(());
            }
        }

        // Execute ingestion
        print_status("Ingesting", "Starting ingestion pipeline...");
        print_separator();

        let file_names: Vec<String> = files_to_process.iter().map(|f| f.name.clone()).collect();
        let request = IngestRequest {
            directory: preview.directory.clone(),
            files: Some(file_names),
        };

        let response = client.ingest(request).await?;

        // Parse task ID from message
        let task_id = extract_task_id(&response.message);
        
        if response.status == "processing" {
            println!("{} {}", "Task ID:".cyan().bold(), task_id.as_deref().unwrap_or("unknown"));
            println!("{} This may take several minutes for large documents.", "Info:".yellow());
            println!();
            
            // Poll for status updates
            if let Some(id) = &task_id {
                poll_status(&client, id, self.format).await?;
            }
        } else {
            // Immediate response (error or warning)
            display_ingest_result(&response, &None, self.format);
        }

        Ok(())
    }
}

/// Extract task ID from response message
fn extract_task_id(message: &str) -> Option<String> {
    // Message format: "Ingestion started in background. N file(s) being processed. Task ID: uuid"
    if let Some(pos) = message.find("Task ID: ") {
        let id_str = &message[pos + 9..];
        // Take until end or whitespace
        id_str.split_whitespace().next().map(|s| s.to_string())
    } else {
        None
    }
}

/// Poll for ingestion status until complete
async fn poll_status(client: &ApiClient, task_id: &str, _format: OutputFormat) -> Result<()> {
    let mut last_status = String::new();
    let mut last_progress = -1;
    
    loop {
        match client.get_ingest_status(task_id).await {
            Ok(status) => {
                let progress = status.progress;
                
                // Only print if status or progress changed
                if status.status != last_status || progress != last_progress {
                    print_status_update(&status, progress, &last_status);
                    last_status = status.status.clone();
                    last_progress = progress;
                }
                
                // Check if complete
                match status.status.as_str() {
                    "completed" => {
                        print_separator();
                        println!("{}", "Ingestion Complete!".green().bold());
                        println!();
                        println!("{} {}", "Nodes extracted:".cyan().bold(), status.total_nodes);
                        println!("{} {}", "Entities found:".cyan().bold(), status.total_entities);
                        println!("{} {}", "Communities built:".cyan().bold(), status.total_communities);
                        println!();
                        
                        if !status.files_processed.is_empty() {
                            println!("{}", "Files processed:".green().bold());
                            for f in &status.files_processed {
                                if f.contains("(FAILED)") {
                                    println!("  {} {}", "✗".red(), f.red());
                                } else {
                                    println!("  {} {}", "✓".green(), f);
                                }
                            }
                        }
                        break;
                    }
                    "error" => {
                        print_separator();
                        print_error(&format!(
                            "Ingestion failed: {}",
                            status.error.unwrap_or_else(|| "Unknown error".to_string())
                        ));
                        break;
                    }
                    _ => {
                        // Still processing, wait before next poll
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    }
                }
            }
            Err(e) => {
                print_error(&format!("Failed to check status: {}", e));
                break;
            }
        }
    }
    
    Ok(())
}

/// Print a status update
fn print_status_update(status: &IngestStatus, progress: i32, _last_status: &str) {
    let status_msg = match status.status.as_str() {
        "queued" => "Queued...".to_string(),
        "extracting_nodes" => format!("Extracting nodes from documents... ({}%)", progress),
        "building_knowledge_graph" => format!("Building knowledge graph... ({}%)", progress),
        "extracting_entities" => format!("Extracting entities with LLM... ({}%)", progress),
        "building_communities" => format!("Building communities... ({}%)", progress),
        "completed" => "Complete!".to_string(),
        "error" => "Error!".to_string(),
        _ => format!("Processing... ({}%)", progress),
    };
    
    println!("{} {}", "→".cyan(), status_msg);
}

/// Display files as a table
fn display_files_table(files: &[IngestFileInfo]) {
    #[derive(Tabled)]
    struct FileRow {
        #[tabled(rename = "#")]
        id: usize,
        #[tabled(rename = "File")]
        name: String,
        #[tabled(rename = "Type")]
        extension: String,
        #[tabled(rename = "Size")]size: String,
    }

    let rows: Vec<FileRow> = files
        .iter()
        .enumerate()
        .map(|(i, f)| FileRow {
            id: i + 1,
            name: f.name.clone(),
            extension: f.extension.clone(),
            size: format_size(f.size_bytes),
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

/// Calculate total size in KB
fn total_size_kb(files: &[IngestFileInfo]) -> f64 {
    files.iter().map(|f| f.size_bytes as f64 / 1024.0).sum()
}

/// Display ingestion result
fn display_ingest_result(response: &IngestResponse, status: &Option<IngestStatus>, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(response).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
            );
        }
        OutputFormat::Yaml => {
            println!("status: {}", response.status);
            println!("directory: {}", response.directory);
            println!("files_processed:");
            for f in &response.files_processed {
                println!("  - {}", f);
            }
            println!("total_nodes: {}", response.total_nodes);
            println!("message: {}", response.message);
            if let Some(s) = status {
                println!("total_entities: {}", s.total_entities);
                println!("total_communities: {}", s.total_communities);
            }
        }
        OutputFormat::Table => {
            print_separator();

            let status_color = if response.status == "success" || response.status == "processing" {
                response.status.green()
            } else if response.status == "error" {
                response.status.red()
            } else {
                response.status.yellow()
            };

            println!("{} {}", "Status:".cyan().bold(), status_color);
            println!("{} {}", "Directory:".cyan().bold(), response.directory);
            println!("{} {}", "Nodes:".cyan().bold(), response.total_nodes);
            
            if let Some(s) = status {
                if s.total_entities > 0 {
                    println!("{} {}", "Entities:".cyan().bold(), s.total_entities);
                }
                if s.total_communities > 0 {
                    println!("{} {}", "Communities:".cyan().bold(), s.total_communities);
                }
            }
            println!();

            if !response.files_processed.is_empty() {
                println!("{}", "Files processed:".green().bold());
                for f in &response.files_processed {
                    if f.contains("(FAILED)") {
                        println!("  {} {}", "✗".red(), f.red());
                    } else {
                        println!("  {} {}", "✓".green(), f);
                    }
                }
            }

            println!();
            println!("{} {}", "Message:".cyan().bold(), response.message);
        }
    }
}

/// Ask user for confirmation
fn confirm(prompt: &str) -> Result<bool> {
    print!("{} ", prompt);
    io::stdout().flush()?;

    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;

    let answer = line.trim().to_lowercase();
    Ok(answer == "y" || answer == "yes")
}