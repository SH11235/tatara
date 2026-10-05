//! 学習成果物を、同じディレクトリの一時ファイルから atomic に公開する。

use std::ffi::OsString;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// 一時ファイル名の末尾。強制終了で残ったファイルをこの suffix で見分ける。
pub const TEMPORARY_SUFFIX: &str = ".tatara-tmp";

const FALLBACK_TEMPORARY_PREFIX: &str = ".tatara-";

// `.` + 出力名 + `.` + 乱数部 + suffix が一般的な上限 255 byte に収まる長さ。
const MAX_ATTRIBUTED_NAME_BYTES: usize = 200;

/// バッファ付きで書き込み、flush と file sync が成功した成果物を置換する。
///
/// 一時ファイルは `.<出力ファイル名>.<乱数>.tatara-tmp` として排他的に確保し、
/// 書き込み・同期・置換の失敗時には削除する。置換前の失敗では既存の出力を
/// 保持する。既存の出力は write permission が無くても置換する。
///
/// Unix では置換後に親ディレクトリを sync する。置換が済んだ時点で成果物は
/// 完全な新ファイルとして公開済みなので、この sync の失敗は warning を出すだけで
/// エラーにはしない。同期非対応のファイルシステムでは warning も出さない。
/// callback のエラー型を保持するため、GPU download の失敗もそのまま伝搬する。
pub fn write_atomic<E>(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<&mut File>) -> Result<(), E>,
) -> Result<(), E>
where
    E: From<io::Error>,
{
    let directory_sync_error = write_atomic_with_directory_sync(path, write, |parent| {
        #[cfg(unix)]
        {
            File::open(parent)?.sync_all()
        }
        #[cfg(not(unix))]
        {
            let _ = parent;
            Ok(())
        }
    })?;
    if let Some(error) = directory_sync_error {
        eprintln!(
            "[artifact] warning: {} was published, but syncing its directory failed: {error}; \
             the new file may not survive a crash or power loss",
            path.display()
        );
    }
    Ok(())
}

/// [`write_atomic`] と同じだが、書き込み用に開けない既存の出力は置換せずエラーにする。
///
/// rename による置換は出力ファイル自体の permission を参照しないため、
/// `chmod a-w` で保護した成果物も上書きできてしまう。出力をその場で開いて
/// 書く場合と同じ保護を保つ出力にはこちらを使う。
pub fn write_atomic_keeping_read_only<E>(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<&mut File>) -> Result<(), E>,
) -> Result<(), E>
where
    E: From<io::Error>,
{
    ensure_destination_writable(path)?;
    write_atomic(path, write)
}

fn ensure_destination_writable(path: &Path) -> io::Result<()> {
    // mode bits の検査では ACL や Windows の read-only 属性を取りこぼすので、
    // truncate なしで実際に開いて判定する。
    match File::options().write(true).open(path) {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn temporary_prefix(path: &Path) -> OsString {
    match path.file_name() {
        Some(name) if name.as_encoded_bytes().len() <= MAX_ATTRIBUTED_NAME_BYTES => {
            let mut prefix = OsString::from(".");
            prefix.push(name);
            prefix.push(".");
            prefix
        }
        _ => OsString::from(FALLBACK_TEMPORARY_PREFIX),
    }
}

/// 置換後のディレクトリ同期が失敗した場合は、その error を `Ok(Some(_))` で返す。
fn write_atomic_with_directory_sync<E>(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<&mut File>) -> Result<(), E>,
    sync_directory: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<Option<io::Error>, E>
where
    E: From<io::Error>,
{
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let prefix = temporary_prefix(path);
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix).suffix(TEMPORARY_SUFFIX);
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
    // Windows の `std::fs::rename` は、置換が拒否されると read-only 属性を無視する
    // POSIX semantics で再試行する。tempfile の persist にはこの再試行が無いため、
    // 自動削除を外した path を `std::fs::rename` に渡し、失敗時の削除は自前で行う。
    let temporary_path = temporary
        .into_temp_path()
        .keep()
        .map_err(|error| error.error)?;
    if let Err(error) = std::fs::rename(&temporary_path, path) {
        let _ = std::fs::remove_file(&temporary_path);
        return Err(error.into());
    }
    match sync_directory(parent) {
        Ok(()) => Ok(None),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::Unsupported | io::ErrorKind::InvalidInput
            ) =>
        {
            Ok(None)
        }
        Err(error) => Ok(Some(error)),
    }
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
        const CHILD_MARKER: &str = "TATARA_PERMISSION_TEST_CHILD_MARKER";
        let Some(marker) = std::env::var_os(CHILD_MARKER) else {
            let marker_directory = tempfile::tempdir()?;
            let marker = marker_directory.path().join("child-ran");
            let status = std::process::Command::new("sh")
                .args(["-c", "umask 077; exec \"$@\"", "sh"])
                .arg(std::env::current_exe()?)
                .args([
                    "--exact",
                    "artifact::tests::replacing_checkpoint_preserves_permissions_under_restrictive_umask",
                    "--nocapture",
                ])
                .env(CHILD_MARKER, &marker)
                .status()?;
            assert!(status.success());
            // filter が何にも一致しない test binary も成功終了するため、child が
            // assertion まで実行した証拠を別途要求する。
            assert!(marker.exists(), "child process did not run the test body");
            return Ok(());
        };
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
        std::fs::write(marker, b"")?;
        Ok(())
    }

    #[test]
    fn temporary_file_is_named_after_its_destination() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("net-100.ckpt");
        write_atomic(&path, |writer| {
            writer.write_all(b"checkpoint")?;
            let names: Vec<String> = std::fs::read_dir(directory.path())?
                .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
                .collect::<io::Result<_>>()?;
            assert_eq!(names.len(), 1, "{names:?}");
            assert!(names[0].starts_with(".net-100.ckpt."), "{names:?}");
            assert!(names[0].ends_with(TEMPORARY_SUFFIX), "{names:?}");
            Ok::<_, io::Error>(())
        })?;
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        Ok(())
    }

    #[test]
    fn overlong_destination_names_fall_back_to_a_generic_temporary_prefix() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let attributed = "a".repeat(MAX_ATTRIBUTED_NAME_BYTES);
        let generic = "a".repeat(240);
        assert_eq!(
            temporary_prefix(Path::new(&attributed)),
            OsString::from(format!(".{attributed}."))
        );
        assert_eq!(
            temporary_prefix(Path::new(&generic)),
            OsString::from(FALLBACK_TEMPORARY_PREFIX)
        );
        for name in [attributed, generic] {
            let path = directory.path().join(name);
            write_atomic(&path, |writer| writer.write_all(b"complete"))?;
            assert_eq!(std::fs::read(&path)?, b"complete");
        }
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 2);
        Ok(())
    }

    /// permission 検査を受けない実行主体 (root 等) では拒否系の test が成立しない。
    #[cfg(unix)]
    fn permission_checks_are_bypassed(directory: &Path) -> io::Result<bool> {
        use std::os::unix::fs::PermissionsExt;
        let probe = directory.join("permission-probe");
        std::fs::write(&probe, b"")?;
        std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o444))?;
        let bypassed = File::options().write(true).open(&probe).is_ok();
        std::fs::remove_file(&probe)?;
        Ok(bypassed)
    }

    #[cfg(unix)]
    #[test]
    fn read_only_destination_is_refused_only_when_keeping_read_only() -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir()?;
        if permission_checks_are_bypassed(directory.path())? {
            return Ok(());
        }
        let path = directory.path().join("net.bin");
        std::fs::write(&path, b"protected")?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444))?;

        let error =
            write_atomic_keeping_read_only(&path, |writer| writer.write_all(b"new")).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(std::fs::read(&path)?, b"protected");
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);

        write_atomic(&path, |writer| writer.write_all(b"replaced"))?;
        assert_eq!(std::fs::read(&path)?, b"replaced");
        assert_eq!(
            std::fs::metadata(&path)?.permissions().mode() & 0o777,
            0o444
        );
        Ok(())
    }

    #[test]
    fn keeping_read_only_still_creates_and_replaces_writable_outputs() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("exports/net.bin");
        write_atomic_keeping_read_only(&path, |writer| writer.write_all(b"first"))?;
        write_atomic_keeping_read_only(&path, |writer| writer.write_all(b"second"))?;
        assert_eq!(std::fs::read(&path)?, b"second");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn non_writable_parent_fails_and_keeps_existing_output() -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir()?;
        if permission_checks_are_bypassed(directory.path())? {
            return Ok(());
        }
        let parent = directory.path().join("frozen");
        std::fs::create_dir(&parent)?;
        let path = parent.join("net.ckpt");
        std::fs::write(&path, b"complete")?;
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o555))?;
        let result = write_atomic(&path, |writer| writer.write_all(b"new"));
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755))?;
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(std::fs::read(&path)?, b"complete");
        assert_eq!(std::fs::read_dir(&parent)?.count(), 1);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_parent_still_publishes_successfully() -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir()?;
        let parent = directory.path().join("write-only");
        std::fs::create_dir(&parent)?;
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o300))?;
        let path = parent.join("net.ckpt");
        let result = write_atomic(&path, |writer| writer.write_all(b"complete"));
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755))?;
        result?;
        assert_eq!(std::fs::read(&path)?, b"complete");
        assert_eq!(std::fs::read_dir(&parent)?.count(), 1);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn other_hard_links_keep_the_previous_contents() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("net.bin");
        let other = directory.path().join("alias.bin");
        std::fs::write(&path, b"old")?;
        std::fs::hard_link(&path, &other)?;
        write_atomic(&path, |writer| writer.write_all(b"new"))?;
        assert_eq!(std::fs::read(&path)?, b"new");
        assert_eq!(std::fs::read(&other)?, b"old");
        Ok(())
    }

    #[test]
    fn unsupported_directory_sync_is_not_reported() -> io::Result<()> {
        for kind in [io::ErrorKind::Unsupported, io::ErrorKind::InvalidInput] {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("net.bin");
            let reported = write_atomic_with_directory_sync(
                &path,
                |writer| writer.write_all(b"complete"),
                |_| Err(io::Error::from(kind)),
            )?;
            assert!(reported.is_none());
            assert_eq!(std::fs::read(&path)?, b"complete");
            assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        }
        Ok(())
    }

    #[test]
    fn other_directory_sync_failures_are_reported_without_failing_publication() -> io::Result<()> {
        for kind in [io::ErrorKind::PermissionDenied, io::ErrorKind::Other] {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("net.bin");
            let reported = write_atomic_with_directory_sync(
                &path,
                |writer| writer.write_all(b"complete"),
                |_| Err(io::Error::from(kind)),
            )?;
            assert_eq!(reported.map(|error| error.kind()), Some(kind));
            assert_eq!(std::fs::read(&path)?, b"complete");
            assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        }
        Ok(())
    }
}
