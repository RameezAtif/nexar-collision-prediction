"""Offline checks: python -m unittest discover -s tools -p 'test_download_smoke_set.py'."""

import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import download_smoke_set as downloader


class DownloadTests(unittest.TestCase):
    def test_nested_flat_and_zipped_downloads(self):
        for layout in ("nested", "flat", "zip"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as temp:
                raw = Path(temp)

                def fake_run(command):
                    staging = Path(command[-1])
                    if layout == "zip":
                        with zipfile.ZipFile(staging / "video.zip", "w") as archive:
                            archive.writestr("train/00000.mp4", b"video contents")
                            archive.writestr("../../outside.txt", b"do not extract")
                    else:
                        target = staging / ("train/00000.mp4" if layout == "nested" else "00000.mp4")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(b"video contents")

                destination = raw / "train/00000.mp4"
                with patch.object(downloader, "RAW", raw), patch.object(downloader, "run", fake_run):
                    downloader.download("kaggle", "train/00000.mp4", destination)
                self.assertEqual(destination.read_bytes(), b"video contents")
                self.assertEqual(list(raw.iterdir()), [raw / "train"])

    def test_failed_transfer_leaves_no_final_file(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)

            def fail(command):
                (Path(command[-1]) / "00000.mp4").write_bytes(b"partial")
                raise subprocess.CalledProcessError(1, command)

            with patch.object(downloader, "RAW", raw), patch.object(downloader, "run", fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    downloader.download("kaggle", "train/00000.mp4", raw / "train/00000.mp4")
            self.assertEqual(list(raw.iterdir()), [])

    def test_completed_set_needs_no_install_login_or_download(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            (raw / "train").mkdir()
            (raw / "train.csv").write_text("existing trimmed labels")
            for video_id in downloader.SMOKE_IDS:
                (raw / "train" / f"{video_id:05d}.mp4").write_bytes(b"existing video")
            with patch.object(downloader, "RAW", raw), patch.object(downloader, "kaggle_command") as setup, \
                    patch("sys.argv", ["download_smoke_set.py"]):
                self.assertEqual(downloader.main(), 0)
                setup.assert_not_called()
            self.assertEqual((raw / "train.csv").read_text(), "existing trimmed labels")


if __name__ == "__main__":
    unittest.main()
