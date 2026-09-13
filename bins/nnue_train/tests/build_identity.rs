use std::path::{Path, PathBuf};
use std::process::Command;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "tatara-build-identity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn run(root: &Path, program: &str, args: &[&str]) -> String {
    let out = Command::new(program)
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{program} {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().into()
}
fn git(root: &Path, args: &[&str]) -> String {
    run(root, "git", args)
}
fn init(root: &Path) {
    git(root, &["init", "-b", "main"]);
    git(root, &["config", "user.email", "test@example.invalid"]);
    git(root, &["config", "user.name", "Build identity test"]);
    git(root, &["config", "commit.gpgsign", "false"]);
}
fn commit(root: &Path) -> String {
    git(root, &["add", "."]);
    git(root, &["commit", "-m", "fixture"]);
    git(root, &["rev-parse", "HEAD"])
}
fn build(root: &Path, target: &Path) -> String {
    let out = Command::new(env!("CARGO"))
        .current_dir(root)
        .args(["build", "--offline", "--quiet", "--target-dir"])
        .arg(target)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    run(
        root,
        target
            .join("debug")
            .join(format!("identity-fixture{}", std::env::consts::EXE_SUFFIX))
            .to_str()
            .unwrap(),
        &[],
    )
}
fn fixture(root: &Path) {
    let package = root.join("bins/nnue_train");
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"bins/nnue_train\"]\nresolver = \"3\"\n",
    )
    .unwrap();
    std::fs::write(root.join(".gitignore"), "Cargo.lock\n").unwrap();
    std::fs::write(
        package.join("Cargo.toml"),
        "[package]\nname = \"identity-fixture\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(package.join("build.rs"), include_str!("../build.rs")).unwrap();
    std::fs::write(
        package.join("build_identity.rs"),
        include_str!("../build_identity.rs"),
    )
    .unwrap();
    std::fs::write(package.join("src/main.rs"), r#"fn main() { println!("{}|{}|{}", env!("TATARA_BUILD_FULL_COMMIT"), env!("TATARA_BUILD_DIRTY"), env!("TATARA_BUILD_COMMIT")); }"#).unwrap();
}

#[test]
fn trainer_identity_is_independent_of_launch_directory() {
    let scratch = Scratch::new();
    let other = scratch.0.join("other");
    std::fs::create_dir(&other).unwrap();
    init(&other);
    std::fs::write(other.join("unrelated"), "other repository").unwrap();
    commit(&other);
    let trainer = env!("CARGO_BIN_EXE_nnue-train");
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let reference = run(manifest, trainer, &["build-info"]);
    assert_eq!(reference, run(&other, trainer, &["build-info"]));
    assert_eq!(reference, run(&scratch.0, trainer, &["build-info"]));
    let identity: serde_json::Value = serde_json::from_str(&reference).unwrap();
    assert!(identity["backend"].is_string());
    assert!(identity["rustc"].as_str().unwrap().starts_with("rustc "));
}

#[test]
fn cargo_refreshes_worktree_commit_refs_and_dirty_state() {
    let scratch = Scratch::new();
    let repo = scratch.0.join("repo");
    std::fs::create_dir(&repo).unwrap();
    fixture(&repo);
    init(&repo);
    let first = commit(&repo);
    let worktree = scratch.0.join("worktree");
    git(
        &repo,
        &["worktree", "add", "-b", "build", worktree.to_str().unwrap()],
    );
    let target = scratch.0.join("target");
    assert!(build(&worktree, &target).starts_with(&format!("{first}|false|")));
    git(&worktree, &["commit", "--allow-empty", "-m", "ref update"]);
    let second = git(&worktree, &["rev-parse", "HEAD"]);
    assert_ne!(first, second);
    assert!(build(&worktree, &target).starts_with(&format!("{second}|false|")));
    git(&worktree, &["pack-refs", "--all"]);
    assert!(build(&worktree, &target).starts_with(&format!("{second}|false|")));
    git(&worktree, &["checkout", "--detach", &first]);
    assert!(build(&worktree, &target).starts_with(&format!("{first}|false|")));
    std::fs::write(worktree.join("change"), "uncommitted").unwrap();
    git(&worktree, &["add", "change"]);
    let dirty = build(&worktree, &target);
    assert!(dirty.starts_with(&format!("{first}|true|")), "{dirty}");
    assert!(dirty.ends_with("-dirty"));
    let third = commit(&worktree);
    assert!(build(&worktree, &target).starts_with(&format!("{third}|false|")));
    let archive = repo.join("archive");
    fixture(&archive);
    assert_eq!(build(&archive, &target), "unknown|unknown|unknown");
    let outside = scratch.0.join("outside");
    fixture(&outside);
    assert_eq!(build(&outside, &target), "unknown|unknown|unknown");
}
