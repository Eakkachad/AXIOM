//! Standalone persistent web-learning worker.

use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::thread;
use std::time::Duration;

use tle_deepman::daemon::WebLearnQueue;
use tle_deepman::web_learning::fetch_html;
use tle_deepman::web_learning::extract_html;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let once = env::args().any(|arg| arg == "--once");
    let queue_path = env::var("AXIOM_WEB_QUEUE").unwrap_or_else(|_| "data/axiom_web_queue.tsv".to_string());
    let output_path = env::var("AXIOM_WEB_OUTPUT").unwrap_or_else(|_| "data/axiom_web_learned.txt".to_string());
    let interval = env::var("AXIOM_WEB_INTERVAL_SECS").ok().and_then(|value| value.parse().ok()).unwrap_or(5);
    let mut queue = WebLearnQueue::load(queue_path);

    loop {
        if let Some(job) = queue.next() {
            match fetch_html(&job.url) {
                Ok(html) => {
                    let page = extract_html(&html);
                    let mut output = OpenOptions::new().create(true).append(true).open(&output_path)?;
                    for sentence in &page.sentences {
                        writeln!(output, "{}", sentence)?;
                    }
                    for fact in &page.facts {
                        writeln!(output, "{} {} {}", fact.subject, fact.relation, fact.object)?;
                    }
                    queue.complete(&job.url)?;
                    println!("learned {}: {} sentences, {} facts", job.url, page.sentences.len(), page.facts.len());
                }
                Err(error) => {
                    let exhausted = queue.fail(&job.url)?;
                    eprintln!("failed {} (exhausted={}): {}", job.url, exhausted, error);
                }
            }
        } else if once {
            break;
        }
        if !once {
            thread::sleep(Duration::from_secs(interval));
        }
    }
    Ok(())
}
