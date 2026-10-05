//! 学習成果物を、同じディレクトリの一時ファイルから atomic に公開する。

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// バッファ付きで書き込み、flush と file sync が成功した成果物を置換する。
///
/// 一時ファイルは排他的に確保し、書き込み・同期・置換の失敗時には削除する。
/// 置換前の失敗では既存の出力を保持する。Unix では置換後に親ディレクトリを
/// sync する。同期非対応のファイルシステムではディレクトリ同期を省略する。
/// その他の最後の同期が失敗した場合も、出力には完全な新ファイルが残る。
/// callback のエラー型を保持するため、GPU download の失敗もそのまま伝搬する。
pub fn write_atomic<E>(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<&mut File>) -> Result<(), E>,
) -> Result<(), E>
where
    E: From<io::Error>,
{
    write_atomic_with_directory_sync(path, write, |parent| {
        #[cfg(unix)]
        {
            File::open(parent)?.sync_all()
        }
        #[cfg(not(unix))]
        {
            let _ = parent;
            Ok(())
        }
    })
}

fn write_atomic_with_directory_sync<E>(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<&mut File>) -> Result<(), E>,
    sync_directory: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<(), E>
where
    E: From<io::Error>,
{
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut builder = tempfile::Builder::new();
    builder.prefix(".tatara-");
    #[cfg(unix)]
    let existing_permissions = {
        use std::os::unix::fs::PermissionsExt;
        let permissions = match std::fs::symlink_metadata(path) {
            Ok(metadata) if metadata.is_file() => Some(metadata.permissions()),
            Ok(_) => None,
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if permissions.is_none() {
            // 新規出力は File::create と同じ umask 制約に従う。
            builder.permissions(std::fs::Permissions::from_mode(0o666));
        }
        permissions
    };
    let mut temporary = builder.tempfile_in(parent)?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        write(&mut writer)?;
        writer.flush()?;
    }
    #[cfg(unix)]
    if let Some(permissions) = existing_permissions {
        // creation の umask で既存ファイルの mode を狭めない。
        temporary.as_file().set_permissions(permissions)?;
    }
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    match sync_directory(parent) {
        Ok(()) => {}
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::Unsupported | io::ErrorKind::InvalidInput
            ) => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn replaces_complete_artifacts_and_creates_parent_directories() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("checkpoints/net.ckpt");
        write_atomic(&path, |writer| writer.write_all(b"first"))?;
        write_atomic(&path, |writer| writer.write_all(b"second"))?;
        assert_eq!(std::fs::read(&path)?, b"second");
        assert_eq!(std::fs::read_dir(path.parent().unwrap())?.count(), 1);
        Ok(())
    }

    #[test]
    fn callback_failure_keeps_existing_output_and_cleans_temporary_file() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("net.bin");
        std::fs::write(&path, b"complete")?;
        let error = write_atomic(&path, |writer| {
            writer.write_all(b"partial")?;
            Err::<(), _>(io::Error::new(
                io::ErrorKind::InvalidData,
                "injected failure",
            ))
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(std::fs::read(&path)?, b"complete");
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        Ok(())
    }

    #[test]
    fn publish_failure_cleans_temporary_file() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("net.bin");
        std::fs::create_dir(&path)?;
        assert!(write_atomic(&path, |writer| writer.write_all(b"checkpoint")).is_err());
        assert!(path.is_dir());
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        Ok(())
    }

    #[test]
    fn concurrent_writers_publish_whole_files_without_sharing_temporary_paths() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("experiment.json");
        let barrier = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            let handles = [b'a', b'b'].map(|value| {
                let path = &path;
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    write_atomic(path, |writer| {
                        writer.write_all(&[value; 128 * 1024])?;
                        barrier.wait();
                        Ok::<_, io::Error>(())
                    })
                })
            });
            for handle in handles {
                handle.join().unwrap()?;
            }
            Ok::<_, io::Error>(())
        })?;
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes.len(), 128 * 1024);
        assert!(bytes.iter().all(|byte| *byte == bytes[0]));
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn never_follows_fixed_temporary_names_or_destination_symlinks() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let unrelated = directory.path().join("unrelated");
        let path = directory.path().join("net.bin");
        std::fs::write(&unrelated, b"untouched")?;
        std::os::unix::fs::symlink(&unrelated, path.with_extension("bin.tmp"))?;
        std::os::unix::fs::symlink(&unrelated, &path)?;
        write_atomic(&path, |writer| writer.write_all(b"checkpoint"))?;
        assert_eq!(std::fs::read(&unrelated)?, b"untouched");
        assert_eq!(std::fs::read(&path)?, b"checkpoint");
        assert!(!path.is_symlink());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn replacing_a_private_checkpoint_does_not_make_it_public() -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("private.ckpt");
        std::fs::write(&path, b"old")?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        write_atomic(&path, |writer| writer.write_all(b"new"))?;
        assert_eq!(
            std::fs::metadata(&path)?.permissions().mode() & 0o777,
            0o600
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn replacing_checkpoint_preserves_permissions_under_restrictive_umask() -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        if std::env::var_os("TATARA_PERMISSION_TEST_CHILD").is_none() {
            let status = std::process::Command::new("sh")
                .args(["-c", "umask 077; exec \"$@\"", "sh"])
                .arg(std::env::current_exe()?)
                .args([
                    "--exact",
                    "artifact::tests::replacing_checkpoint_preserves_permissions_under_restrictive_umask",
                    "--nocapture",
                ])
                .env("TATARA_PERMISSION_TEST_CHILD", "1")
                .status()?;
            assert!(status.success());
            return Ok(());
        }
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("net.bin");
        std::fs::write(&path, b"old")?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))?;
        write_atomic(&path, |writer| writer.write_all(b"new"))?;
        assert_eq!(
            std::fs::metadata(&path)?.permissions().mode() & 0o777,
            0o644
        );
        let new_path = directory.path().join("new.bin");
        write_atomic(&new_path, |writer| writer.write_all(b"new"))?;
        assert_eq!(
            std::fs::metadata(&new_path)?.permissions().mode() & 0o777,
            0o600
        );
        Ok(())
    }

    #[test]
    fn unsupported_directory_sync_keeps_publication_successful() -> io::Result<()> {
        for kind in [io::ErrorKind::Unsupported, io::ErrorKind::InvalidInput] {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("net.bin");
            write_atomic_with_directory_sync(
                &path,
                |writer| writer.write_all(b"complete"),
                |_| Err(io::Error::from(kind)),
            )?;
            assert_eq!(std::fs::read(&path)?, b"complete");
            assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        }
        Ok(())
    }

    #[test]
    fn other_directory_sync_failures_propagate_after_complete_publication() -> io::Result<()> {
        for kind in [io::ErrorKind::PermissionDenied, io::ErrorKind::Other] {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("net.bin");
            let error = write_atomic_with_directory_sync(
                &path,
                |writer| writer.write_all(b"complete"),
                |_| Err(io::Error::from(kind)),
            )
            .unwrap_err();
            assert_eq!(error.kind(), kind);
            assert_eq!(std::fs::read(&path)?, b"complete");
            assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        }
        Ok(())
    }
}
