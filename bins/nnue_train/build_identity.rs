use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub struct GitIdentity {
    pub commit: Option<String>,
    pub short_commit: Option<String>,
    pub dirty: Option<bool>,
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8(out.stdout).ok())
        .flatten()
        .map(|s| s.trim().to_owned())
}

pub fn capture(root: &Path) -> GitIdentity {
    let unknown = || GitIdentity {
        commit: None,
        short_commit: None,
        dirty: None,
    };
    // An unpacked source tree inside another repository must not inherit its revision.
    let Some(top) = git(root, &["rev-parse", "--show-toplevel"]) else {
        return unknown();
    };
    if std::fs::canonicalize(top).ok() != std::fs::canonicalize(root).ok() {
        return unknown();
    }
    let commit = git(root, &["rev-parse", "--verify", "HEAD"]);
    if commit.as_ref().is_none_or(String::is_empty) {
        return unknown();
    }
    GitIdentity {
        commit,
        short_commit: git(root, &["rev-parse", "--short", "HEAD"]),
        dirty: git(root, &["status", "--porcelain", "--untracked-files=normal"])
            .map(|s| !s.is_empty()),
    }
}

pub fn rerun_paths(root: &Path) -> Vec<PathBuf> {
    let mut paths = vec![root.join(".git")];
    if let Some(dir) = git(root, &["rev-parse", "--absolute-git-dir"]) {
        paths.push(Path::new(&dir).join("HEAD"));
        paths.push(Path::new(&dir).join("index"));
    }
    if let Some(common) = git(
        root,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    ) {
        paths.push(Path::new(&common).join("refs"));
        paths.push(Path::new(&common).join("packed-refs"));
    }
    paths
}
