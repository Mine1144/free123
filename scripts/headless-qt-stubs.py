"""Run the Qt UI tests of this project inside a container that has no X server or libGL.

Sandbox/dev helper only — nothing here ships with the Windows app. PySide6 needs a handful of
system libraries that minimal images do not have (libGL, libEGL, libxkbcommon, libxcb, libdbus).
The script scans every shared object of the installed Python packages *and* the system libraries
they pull in for undefined symbols, then builds a stub for each missing SONAME containing exactly
those symbols. Two details make or break this on GNU ld 2.40:

* version script entries need a trailing ``;`` per name, and a *versioned* definition does not
  satisfy a consumer that asks for the bare name (libX11 -> xcb_get_file_descriptor), so the
  stub is built unversioned when possible;
* dependencies matter: librsvg/libX11 pull in symbols that are not referenced by Qt itself.

Usage:
    python3 scripts/headless-qt-stubs.py /tmp/qtstubs
    QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH=/tmp/qtstubs python -m pytest -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# missing library -> (sonames to emit, symbol-name prefixes that belong to it)
TARGETS = {
    "gl": (["libGL.so.1", "libEGL.so.1", "libGLX.so.0"], ("gl", "egl", "GL", "glX", "glapi")),
    "xkb": (["libxkbcommon.so.0"], ("xkb",)),
    "xcb": (["libxcb.so.1"], ("xcb_",)),
    "dbus": (["libdbus-1.so.3"], ("dbus_",)),
}
SKIP_VERSIONS = ("GLIBC", "GCC", "CXXABI", "GLIBCXX", "QT")


def consumers() -> list[Path]:
    roots = []
    try:
        import PySide6
        roots.append(Path(PySide6.__file__).parent)
    except ImportError:
        pass
    for name in ("cv2", "mediapipe", "rapidocr_onnxruntime", "onnxruntime"):
        try:
            module = __import__(name)
            roots.append(Path(module.__file__).parent)
        except ImportError:
            continue
    paths = []
    for root in roots:
        for path in root.rglob("*.so*"):
            if path.is_file() and not path.is_symlink():
                paths.append(path)
    return paths


def search_dirs() -> list[Path]:
    dirs = [Path("/lib/x86_64-linux-gnu"), Path("/usr/lib/x86_64-linux-gnu"),
            Path("/lib64"), Path("/usr/lib64"), Path("/usr/lib"), Path("/lib")]
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if entry.strip():
            dirs.append(Path(entry))
    return [path for path in dirs if path.is_dir()]


def needed_sonames(path: Path) -> list[str]:
    try:
        output = subprocess.run(["readelf", "-d", "--wide", str(path)],
                                capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.split("[", 1)[-1].rstrip("]").strip()
            for line in output.splitlines() if "(NEEDED)" in line]


def dependency_files(roots: list[Path], depth: int = 3) -> list[Path]:
    dirs = search_dirs()
    seen = {path.resolve() for path in roots}
    frontier = list(roots)
    for _ in range(max(1, depth)):
        following = []
        for path in frontier:
            for soname in needed_sonames(path):
                target = next((directory / soname for directory in dirs
                               if (directory / soname).exists()), None)
                if target is None or target.resolve() in seen:
                    continue
                seen.add(target.resolve())
                following.append(target)
        frontier = following
        if not frontier:
            break
    return [Path(path) for path in seen]


def undefined_symbols(path: Path) -> list[tuple[str, str]]:
    try:
        output = subprocess.run(["readelf", "--dyn-syms", "--wide", str(path)],
                                capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in output.splitlines():
        fields = line.split()
        if "UND" not in fields:
            continue
        index = fields.index("UND")
        if index + 1 >= len(fields):
            continue
        symbol, _, version = fields[index + 1].partition("@")
        if symbol.strip():
            found.append((symbol.strip(), version.strip("@")))
    return found


def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qtstubs")
    outdir.mkdir(parents=True, exist_ok=True)
    paths = dependency_files(consumers())
    print(f"scanning {len(paths)} shared objects (consumers + dependencies)")

    symbols: set[tuple[str, str]] = set()
    for path in paths:
        symbols.update(undefined_symbols(path))

    for key, (sonames, prefixes) in TARGETS.items():
        versions: dict[str, set[str]] = {}
        for symbol, version in symbols:
            if version.startswith(SKIP_VERSIONS):
                continue
            if any(symbol.startswith(prefix) for prefix in prefixes):
                versions.setdefault(version, set()).add(symbol)
        names = sorted({name for group in versions.values() for name in group})
        if not names:
            continue
        source = outdir / f"{key}.c"
        source.write_text("/* generated by scripts/headless-qt-stubs.py */\n"
                          + "\n".join(f"long {name}(void) {{ return 0; }}" for name in names) + "\n")
        versioned = sorted(v for v in versions if v)
        script = None
        if versioned:
            script = outdir / f"{key}.map"
            lines = []
            for version in versioned:
                body = " ".join(name + ";" for name in sorted(versions[version]))
                lines.append(f"{version} {{ global: {body} }};")
            script.write_text("\n".join(lines) + "\n")
        for soname in sonames:
            command = ["cc", "-w", "-shared", "-fPIC", "-o", str(outdir / soname), str(source),
                       f"-Wl,-soname,{soname}"]
            if script is not None:
                command.append(f"-Wl,--version-script={script}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FAILED {soname}: {result.stderr.strip()[:300]}")
                continue
            kind = f"{len(versioned)} version group(s)" if versioned else "unversioned"
            print(f"built {soname}: {len(names)} symbols, {kind}")
    print("done:", ", ".join(sorted(path.name for path in outdir.glob("lib*.so*"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
