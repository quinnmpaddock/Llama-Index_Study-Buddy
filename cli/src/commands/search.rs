use crate::api::ApiClient;
use crate::config::Settings;
use crate::error::Result;
use crate::output::{OutputFormat, print_error, print_status};

/// Search command for finding entities in the knowledge graph
pub struct SearchCommand {
    query: Option<String>,
    entity_name: Option<String>,
    limit: i32,
    format: OutputFormat,
}

impl SearchCommand {
    pub fn new() -> Self {
        Self {
            query: None,
            entity_name: None,
            limit: 50,
            format: OutputFormat::Table,
        }
    }

    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    pub fn with_entity(mut self, name: impl Into<String>) -> Self {
        self.entity_name = Some(name.into());
        self
    }

    pub fn with_limit(mut self, limit: i32) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    pub async fn execute(&self, settings: &Settings) -> Result<()> {
        let client = ApiClient::new(settings)?;

        // If entity_name is provided, get specific entity details
        if let Some(ref name) = self.entity_name {
            self.show_entity_details(&client, name).await?;
        } else {
            // Otherwise, search for entities
            self.search_entities(&client).await?;
        }

        Ok(())
    }

    async fn search_entities(&self, client: &ApiClient) -> Result<()> {
        print_status("Searching", &format!(
            "entities matching '{}' (limit: {})...",
            self.query.as_deref().unwrap_or("*"),
            self.limit
        ));

        let response = client.search_entities(self.query.as_deref(), self.limit).await?;

        if response.entities.is_empty() {
            print_error("No entities found.");
            return Ok(());
        }

        match self.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&response).unwrap());
            }
            OutputFormat::Yaml => {
                println!("{}", serde_yaml::to_string(&response).unwrap());
            }
            OutputFormat::Table => {
                use tabled::{Table, Tabled, settings::Style};
                
                #[derive(Tabled)]
                struct EntityRow {
                    #[tabled(rename = "ENTITY")]
                    name: String,
                    #[tabled(rename = "COMMUNITIES")]
                    communities: String,
                }

                let rows: Vec<EntityRow> = response
                    .entities
                    .iter()
                    .map(|e| EntityRow {
                        name: e.name.clone(),
                        communities: e
                            .communities
                            .iter()
                            .map(|c| c.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    })
                    .collect();

                let table = Table::new(rows).with(Style::rounded()).to_string();
                println!("{}", table);
                println!();
                print_status("Results", &format!("{} entities found (total: {})", 
                    response.entities.len(), response.total));
            }
        }

        Ok(())
    }

    async fn show_entity_details(&self, client: &ApiClient, name: &str) -> Result<()> {
        print_status("Fetching", &format!("entity '{}'...", name));

        let entity = client.get_entity(name).await?;

        match self.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&entity).unwrap());
            }
            OutputFormat::Yaml => {
                println!("{}", serde_yaml::to_string(&entity).unwrap());
            }
            OutputFormat::Table => {
                use colored::Colorize;
                
                println!();
                println!("{}", entity.name.bold().underline());
                println!();
                println!("Communities:");
                for comm in &entity.communities {
                    println!("  {}. Community {}", "•".cyan(), comm);
                }
                println!();
            }
        }

        Ok(())
    }
}