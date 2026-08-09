//! Persistent background web-learning queue.

use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebLearnJob {
    pub url: String,
    pub attempts: u32,
}

#[derive(Debug)]
pub struct WebLearnQueue {
    path: String,
    jobs: Vec<WebLearnJob>,
    max_attempts: u32,
}

impl WebLearnQueue {
    pub fn load(path: impl Into<String>) -> Self {
        let path = path.into();
        let jobs = fs::read_to_string(&path)
            .ok()
            .map(|data| data.lines().filter_map(parse_job).collect())
            .unwrap_or_default();
        Self { path, jobs, max_attempts: 3 }
    }

    pub fn enqueue(&mut self, url: &str) -> Result<(), String> {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err("URL must use http:// or https://".to_string());
        }
        if !self.jobs.iter().any(|job| job.url == url) {
            self.jobs.push(WebLearnJob { url: url.to_string(), attempts: 0 });
            self.persist()?;
        }
        Ok(())
    }

    pub fn next(&mut self) -> Option<WebLearnJob> {
        self.jobs.first().cloned()
    }

    pub fn complete(&mut self, url: &str) -> Result<(), String> {
        self.jobs.retain(|job| job.url != url);
        self.persist()
    }

    pub fn fail(&mut self, url: &str) -> Result<bool, String> {
        if let Some(job) = self.jobs.iter_mut().find(|job| job.url == url) {
            job.attempts += 1;
            let exhausted = job.attempts >= self.max_attempts;
            if exhausted {
                self.jobs.retain(|candidate| candidate.url != url);
            }
            self.persist()?;
            return Ok(exhausted);
        }
        Ok(false)
    }

    pub fn len(&self) -> usize { self.jobs.len() }
    pub fn is_empty(&self) -> bool { self.jobs.is_empty() }

    fn persist(&self) -> Result<(), String> {
        if let Some(parent) = Path::new(&self.path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
        }
        let data = self.jobs.iter()
            .map(|job| format!("{}\t{}", job.attempts, job.url.replace('\n', "")))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&self.path, data).map_err(|error| error.to_string())
    }
}

fn parse_job(line: &str) -> Option<WebLearnJob> {
    let (attempts, url) = line.split_once('\t')?;
    Some(WebLearnJob { attempts: attempts.parse().ok()?, url: url.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_persists_and_retries_jobs() {
        let path = std::env::temp_dir().join(format!("axiom_queue_{}", std::process::id()));
        let path_string = path.to_string_lossy().to_string();
        let mut queue = WebLearnQueue::load(&path_string);
        queue.enqueue("https://example.com").unwrap();
        assert_eq!(queue.next().unwrap().attempts, 0);
        assert!(!queue.fail("https://example.com").unwrap());
        let mut restored = WebLearnQueue::load(&path_string);
        assert_eq!(restored.next().unwrap().attempts, 1);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn invalid_urls_are_rejected() {
        let mut queue = WebLearnQueue::load("/tmp/axiom-invalid-queue");
        assert!(queue.enqueue("ftp://example.com").is_err());
    }
}
