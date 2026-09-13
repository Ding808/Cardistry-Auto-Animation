"""Actual Windows sharing-handle checks; no reconstruction, GPU or UE process."""
import ctypes
from ctypes import wintypes
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from cardcap import atomic_file
from cardcap.export import atomic_json as observation_json
from cardcap.editor_job import atomic_json as job_json

MEASUREMENTS = []


def block_delete_sharing(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ; FILE_SHARE_READ | FILE_SHARE_WRITE, deliberately not DELETE.
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return lambda: kernel.CloseHandle(handle)


@unittest.skipUnless(os.name == "nt", "Requires actual Windows CreateFileW sharing semantics")
class AtomicProgressTests(unittest.TestCase):
    writers = (observation_json, job_json)

    def test_temporary_reader_lock_recovers_with_complete_json(self):
        for writer in self.writers:
            with self.subTest(writer=writer.__module__), tempfile.TemporaryDirectory(prefix="atomic 占用 ") as directory:
                path = Path(directory) / "progress.json"
                writer(path, {"status": "running", "frames_completed": 140})
                close_handle = block_delete_sharing(path)
                first_failure = threading.Event()
                release_errors = []
                errors = []
                original = os.replace

                def attempt(source, target):
                    try:
                        return original(source, target)
                    except OSError as error:
                        errors.append(error.winerror)
                        first_failure.set()
                        raise

                def release():
                    if not first_failure.wait(2):
                        release_errors.append("Replacement never encountered real sharing lock")
                    time.sleep(.12)
                    if not close_handle():
                        release_errors.append("CloseHandle failed")

                thread = threading.Thread(target=release)
                thread.start()
                start = time.perf_counter()
                try:
                    with mock.patch.object(atomic_file.os, "replace", side_effect=attempt) as replace:
                        writer(path, {"status": "running", "frames_completed": 141, "message": "完整进度"})
                finally:
                    thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertFalse(release_errors)
                self.assertTrue(errors)
                self.assertTrue(set(errors).issubset({5, 32, 33}))
                self.assertGreater(replace.call_count, 1)
                self.assertLessEqual(replace.call_count, 21)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                                 {"status": "running", "frames_completed": 141, "message": "完整进度"})
                self.assertFalse(path.with_name(path.name + ".tmp").exists())
                MEASUREMENTS.append({"case": "real_temporary_reader", "writer": writer.__module__,
                                     "attempts": replace.call_count, "winerrors": errors,
                                     "elapsed_seconds": time.perf_counter() - start})

    def test_persistent_reader_lock_raises_and_preserves_old_complete_json(self):
        for writer in self.writers:
            with self.subTest(writer=writer.__module__), tempfile.TemporaryDirectory(prefix="atomic 占用 ") as directory:
                path = Path(directory) / "status.json"
                writer(path, {"status": "running", "frames_completed": 140})
                old = path.read_bytes()
                close_handle = block_delete_sharing(path)
                start = time.perf_counter()
                try:
                    with mock.patch.object(atomic_file.os, "replace", wraps=os.replace) as replace:
                        with self.assertRaises(OSError) as raised:
                            writer(path, {"status": "succeeded", "frames_completed": 452})
                    self.assertIn(raised.exception.winerror, {5, 32, 33})
                    self.assertEqual(replace.call_count, 21)
                    self.assertEqual(path.read_bytes(), old)
                    self.assertEqual(json.loads(path.read_text())["status"], "running")
                    self.assertFalse(path.with_name(path.name + ".tmp").exists())
                    MEASUREMENTS.append({"case": "real_persistent_reader", "writer": writer.__module__,
                                         "attempts": replace.call_count, "winerror": raised.exception.winerror,
                                         "elapsed_seconds": time.perf_counter() - start,
                                         "old_bytes_preserved": True, "unpublished_temporary_removed": True})
                finally:
                    close_handle()

    def test_unrelated_errors_do_not_retry_or_publish_success(self):
        for writer in self.writers:
            # Actual WinError construction; OS failure itself is injected here.
            for error in (ctypes.WinError(87), ctypes.WinError(112), PermissionError(errno.EACCES, "no WinError")):
                with self.subTest(writer=writer.__module__, error=str(error)), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "status.json"
                    writer(path, {"status": "running"})
                    old = path.read_bytes()
                    with mock.patch.object(atomic_file.os, "replace", side_effect=error) as replace:
                        with mock.patch.object(atomic_file.time, "sleep") as sleep:
                            with self.assertRaises(OSError) as raised:
                                writer(path, {"status": "succeeded"})
                    self.assertIs(raised.exception, error)
                    self.assertEqual(replace.call_count, 1)
                    sleep.assert_not_called()
                    self.assertEqual(path.read_bytes(), old)
                    self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
