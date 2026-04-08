use thiserror::Error;

#[derive(Debug, Error)]
pub enum StudyBuddyError {
    #[error("API connection failed: {0}")]
    ApiConnection(#[from] reqwest::Error),

    #[error("API error ({status}): {message}")]
    ApiError { status: u16, message: String },

    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, StudyBuddyError>;