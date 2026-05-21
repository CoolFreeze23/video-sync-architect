#!/usr/bin/env python3
"""Batch merge pairs of SRT subtitle files into styled ASS files (GUI)."""

import re
import os
import sys
import json
import subprocess
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
from difflib import SequenceMatcher

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "batch_2srt2ass_config.json"

# ---------------------------------------------------------------------------
# ASS template parsing
# ---------------------------------------------------------------------------

def parse_ass_template(path: Path) -> tuple[list[str], list[str]]:
    """Extract [Script Info] and [V4+ Styles] sections from a .ass file.

    Returns (script_info_lines, styles_lines) with section headers included.
    Skips [Aegisub Project Garbage].
    """
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    script_info: list[str] = []
    styles: list[str] = []
    current: list[str] | None = None

    for line in lines:
        stripped = line.strip()
        if stripped == "[Script Info]":
            current = script_info
            current.append(stripped)
            continue
        if stripped == "[V4+ Styles]":
            current = styles
            current.append(stripped)
            continue
        if stripped.startswith("["):
            current = None
            continue
        if current is not None and stripped:
            if stripped.startswith(";"):
                continue
            current.append(stripped)

    return script_info, styles

# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

SRT_BLOCK = re.compile(
    r"(\d+)\s*\r?\n"
    r"(\d{2}:\d{2}:\d{2})[.,](\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2})[.,](\d{3})\s*\r?\n"
    r"((?:(?!\r?\n\r?\n).)*)",
    re.DOTALL,
)

HTML_TAG = re.compile(r"<[^>]+>")


def srt_ts_to_ass(hms: str, ms: str) -> str:
    """Convert SRT timestamp parts to ASS format (H:MM:SS.cc)."""
    h, m, s = hms.split(":")
    centiseconds = int(ms[:2]) if len(ms) >= 2 else int(ms) * 10
    return f"{int(h)}:{m}:{s}.{centiseconds:02d}"


def parse_srt(path: Path) -> list[dict]:
    """Parse an SRT file and return a list of cue dicts."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    text = text.strip() + "\n\n"
    cues = []
    for m in SRT_BLOCK.finditer(text):
        start = srt_ts_to_ass(m.group(2), m.group(3))
        end = srt_ts_to_ass(m.group(4), m.group(5))
        body = m.group(6).strip()
        body = HTML_TAG.sub("", body)
        body = body.replace("\r\n", "\n").replace("\r", "\n")
        body = re.sub(r"\n", r"\\N", body)
        cues.append({"start": start, "end": end, "text": body})
    return cues

# ---------------------------------------------------------------------------
# ASS generation
# ---------------------------------------------------------------------------

TRACK_TITLE = "English + Portuguese For Julia <3"

def build_ass(
    script_info: list[str],
    styles: list[str],
    top_cues: list[dict],
    bot_cues: list[dict],
) -> str:
    """Build a complete ASS file string from template sections and cues."""
    lines: list[str] = []

    for l in script_info:
        if l.startswith("Title:"):
            continue
        lines.append(l)
    lines.insert(1, f"Title: {TRACK_TITLE}")
    lines.insert(2, "Language: pt")
    lines.append("")

    for l in styles:
        lines.append(l)
    lines.append("")

    lines.append("[Events]")
    lines.append(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )

    events = []
    for c in top_cues:
        events.append((c["start"], "Top", c))
    for c in bot_cues:
        events.append((c["start"], "Bot", c))

    events.sort(key=lambda e: e[0])

    for _ts, style, cue in events:
        lines.append(
            f"Dialogue: 0,{cue['start']},{cue['end']},{style},,0,0,0,,{cue['text']}"
        )

    return "\r\n".join(lines) + "\r\n"

# ---------------------------------------------------------------------------
# File matching
# ---------------------------------------------------------------------------

LANG_SUFFIXES = re.compile(
    r"[._-](?:en|eng|english|pt|por|pt-br|ptbr|portuguese|es|esp|spa|spanish|"
    r"fr|fre|fra|french|de|deu|ger|german|it|ita|italian|ja|jpn|japanese|"
    r"ko|kor|korean|zh|chi|zho|chinese|ru|rus|russian|ar|ara|arabic|"
    r"hi|hin|hindi|tr|tur|turkish|pl|pol|polish|nl|dut|nld|dutch|"
    r"sv|swe|swedish|da|dan|danish|no|nor|norwegian|fi|fin|finnish|"
    r"cs|cze|ces|czech|hu|hun|hungarian|ro|ron|rum|romanian|"
    r"th|tha|thai|vi|vie|vietnamese|id|ind|indonesian|ms|msa|malay|"
    r"uk|ukr|ukrainian|el|gre|ell|greek|he|heb|hebrew|"
    r"sdh|cc|forced|full|default|hearing.impaired|hi)"
    r"$",
    re.IGNORECASE,
)


def base_name(filename: str) -> str:
    """Strip language suffix and extension to get a comparable base name."""
    stem = Path(filename).stem
    stem = LANG_SUFFIXES.sub("", stem)
    stem = LANG_SUFFIXES.sub("", stem)
    return stem.lower().strip()


def match_files(
    top_files: list[str], bot_files: list[str]
) -> tuple[list[tuple[str, str, float]], list[str], list[str]]:
    """Auto-match Top and Bot SRT files by filename similarity.

    Returns (pairs, unmatched_top, unmatched_bot) where each pair is
    (top_filename, bot_filename, confidence).
    """
    top_bases = {f: base_name(f) for f in top_files}
    bot_bases = {f: base_name(f) for f in bot_files}

    pairs: list[tuple[str, str, float]] = []
    used_bot: set[str] = set()

    for tf in top_files:
        best_score = 0.0
        best_bf = ""
        for bf in bot_files:
            if bf in used_bot:
                continue
            if top_bases[tf] == bot_bases[bf]:
                score = 1.0
            else:
                score = SequenceMatcher(None, top_bases[tf], bot_bases[bf]).ratio()
            if score > best_score:
                best_score = score
                best_bf = bf
        if best_score >= 0.4 and best_bf:
            pairs.append((tf, best_bf, best_score))
            used_bot.add(best_bf)

    unmatched_top = [f for f in top_files if not any(p[0] == f for p in pairs)]
    unmatched_bot = [f for f in bot_files if f not in used_bot]

    pairs.sort(key=lambda p: p[0].lower())
    return pairs, unmatched_top, unmatched_bot

PLEX_TRACK_LABEL = "English + Portuguese For Julia"

def output_name(top_srt: str) -> str:
    """Derive output .ass filename in Plex-compatible format.

    Plex reads: MovieName.LanguageCode.Description.ass
    """
    stem = Path(top_srt).stem
    cleaned = LANG_SUFFIXES.sub("", stem)
    cleaned = LANG_SUFFIXES.sub("", cleaned)
    return f"{cleaned}.pt.{PLEX_TRACK_LABEL}.ass"

# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def find_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_path, ffprobe_path) or raise if not found."""
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name):
            continue
        fallback = Path(r"C:\ffmpeg\bin") / f"{name}.exe"
        if fallback.is_file():
            os.environ["PATH"] = str(fallback.parent) + os.pathsep + os.environ.get("PATH", "")
        else:
            raise FileNotFoundError(
                f"{name} not found on PATH or at {fallback}.\n"
                "Install FFmpeg and make sure it's on your PATH."
            )
    return (shutil.which("ffmpeg") or "ffmpeg",
            shutil.which("ffprobe") or "ffprobe")


def probe_subtitles(mkv_path: Path) -> list[dict]:
    """Return a list of subtitle stream dicts from an MKV file.

    Each dict has keys: index (int), codec, language, title.
    """
    _, ffprobe = find_ffmpeg()
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-select_streams", "s",
        str(mkv_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")

    data = json.loads(result.stdout)
    tracks: list[dict] = []
    for s in data.get("streams", []):
        tags = s.get("tags", {})
        tracks.append({
            "index": s["index"],
            "codec": s.get("codec_name", "?"),
            "language": tags.get("language", "und"),
            "title": tags.get("title", ""),
        })
    return tracks


def extract_subtitle(mkv_path: Path, stream_index: int, out_path: Path,
                     codec: str = "srt") -> Path:
    """Extract a single subtitle stream from an MKV.

    If the source codec is ASS/SSA, extracts as-is (copy). Otherwise
    converts to SRT.
    """
    ffmpeg, _ = find_ffmpeg()
    is_ass = codec.lower() in ("ass", "ssa")
    if is_ass:
        out_path = out_path.with_suffix(".ass")
    cmd = [
        ffmpeg, "-y", "-v", "quiet",
        "-i", str(mkv_path),
        "-map", f"0:{stream_index}",
        "-c:s", "copy" if is_ass else "srt",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed:\n{result.stderr}")
    return out_path


ENG_TAGS = {"en", "eng", "english"}
POR_TAGS = {"pt", "por", "portuguese"}


def auto_pick_tracks(tracks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition tracks into (english_candidates, portuguese_candidates)."""
    eng = [t for t in tracks if t["language"].lower() in ENG_TAGS]
    por = [t for t in tracks if t["language"].lower() in POR_TAGS]
    return eng, por

# ---------------------------------------------------------------------------
# ASS-to-ASS merge helpers
# ---------------------------------------------------------------------------

POS_TAG_RE = re.compile(r"\\(?:pos|move)\([^)]*\)")
LAYER_RE = re.compile(r"^(Dialogue|Comment):\s*(\d+)(,.*)$")
ENG_LAYER_OFFSET = 10


POS_XY_RE = re.compile(r"\\pos\(([^,]+),([^)]+)\)")
MOVE_XY_RE = re.compile(r"\\move\(([^,]+),([^,]+),([^,]+),([^,)]+)")


def _offset_pos_y(raw_line: str, y_offset: int) -> str:
    """Shift Y in \\pos(x,y) and \\move(x1,y1,x2,y2,...) by *y_offset* pixels."""
    def shift_pos(m):
        x, y = m.group(1), m.group(2)
        try:
            return f"\\pos({x},{int(float(y)) + y_offset})"
        except ValueError:
            return m.group(0)

    def shift_move(m):
        x1, y1, x2, y2 = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            new_y1 = int(float(y1)) + y_offset
            new_y2 = int(float(y2)) + y_offset
            return f"\\move({x1},{new_y1},{x2},{new_y2}"
        except ValueError:
            return m.group(0)

    raw_line = POS_XY_RE.sub(shift_pos, raw_line)
    raw_line = MOVE_XY_RE.sub(shift_move, raw_line)
    return raw_line


FS_OVERRIDE_RE = re.compile(r"\\fs(\d+(?:\.\d+)?)")


def _ass_time_to_cs(ts: str) -> int:
    """Convert ASS timestamp H:MM:SS.cc to centiseconds."""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s_parts = parts[2].split(".")
    s, cs = int(s_parts[0]), int(s_parts[1])
    return h * 360000 + m * 6000 + s * 100 + cs


def _parse_event_times(raw_line: str) -> tuple[str, str] | None:
    """Extract (start, end) timestamps from a Dialogue/Comment line."""
    parts = raw_line.split(",", 3)
    if len(parts) >= 3:
        return parts[1].strip(), parts[2].strip()
    return None


def _get_event_pos_y(raw_line: str) -> int | None:
    """Extract Y from \\pos(x,y) in an event line."""
    m = POS_XY_RE.search(raw_line)
    if m:
        try:
            return int(float(m.group(2)))
        except ValueError:
            pass
    return None


def _get_event_font_size(raw_line: str, style_font_map: dict[str, int]) -> int:
    """Get effective font size: \\fs override wins, otherwise style default."""
    m = FS_OVERRIDE_RE.search(raw_line)
    if m:
        try:
            return int(float(m.group(1)))
        except ValueError:
            pass
    parts = raw_line.split(",", 4)
    if len(parts) >= 5:
        style = parts[3].strip()
        return style_font_map.get(style, 20)
    return 20


def _build_style_font_map(style_lines: list[str]) -> dict[str, int]:
    """Build a map of style_name -> font_size from style lines."""
    font_map: dict[str, int] = {}
    for line in style_lines:
        if not line.startswith("Style:"):
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            name = parts[0].replace("Style:", "").strip()
            try:
                font_map[name] = int(float(parts[2].strip()))
            except ValueError:
                pass
    return font_map


def _compute_sign_offsets(
    eng_events: list[tuple[str, str]],
    por_events: list[tuple[str, str]],
    style_font_map: dict[str, int],
) -> dict[int, int]:
    """For each Portuguese sign event, compute how many pixels to shift Y.

    Returns a dict mapping por_event_index -> y_offset.  Only events
    whose \\pos() is vertically close to an overlapping English sign
    get an entry (others need no adjustment).
    """
    eng_signs: list[tuple[int, int, int, int]] = []
    for _ts, raw in eng_events:
        parts = raw.split(",", 4)
        if len(parts) < 5:
            continue
        style = parts[3].strip()
        if _is_dialogue_style(style):
            continue
        pos_y = _get_event_pos_y(raw)
        times = _parse_event_times(raw)
        if pos_y is None or times is None:
            continue
        fs = _get_event_font_size(raw, style_font_map)
        eng_signs.append((
            _ass_time_to_cs(times[0]),
            _ass_time_to_cs(times[1]),
            pos_y,
            fs,
        ))

    offsets: dict[int, int] = {}
    for idx, (_ts, raw) in enumerate(por_events):
        parts = raw.split(",", 4)
        if len(parts) < 5:
            continue
        style = parts[3].strip()
        if _is_dialogue_style(style):
            continue
        por_y = _get_event_pos_y(raw)
        times = _parse_event_times(raw)
        if por_y is None or times is None:
            continue
        por_fs = _get_event_font_size(raw, style_font_map)
        por_start = _ass_time_to_cs(times[0])
        por_end = _ass_time_to_cs(times[1])

        best_eng_bottom = 0
        for eng_s, eng_e, eng_y, eng_fs in eng_signs:
            if por_start >= eng_e or por_end <= eng_s:
                continue
            if abs(por_y - eng_y) > SIGN_PROXIMITY_PX:
                continue
            eng_bottom = eng_y + eng_fs + POR_BOT_GAP
            best_eng_bottom = max(best_eng_bottom, eng_bottom)

        if best_eng_bottom > 0 and por_y < best_eng_bottom:
            offsets[idx] = best_eng_bottom - por_y

    return offsets


def _bump_event_layer(raw_line: str, offset: int) -> str:
    """Add *offset* to the Layer field of a Dialogue/Comment event line.

    Different layers prevent the renderer from applying collision logic
    between English and Portuguese events, so each stays at its exact
    MarginV position.
    """
    m = LAYER_RE.match(raw_line)
    if m:
        old_layer = int(m.group(2))
        return f"{m.group(1)}: {old_layer + offset}{m.group(3)}"
    return raw_line


def rename_ass_styles(text: str, suffix: str = "_BR") -> str:
    """Append *suffix* to every style name in an ASS file's text.

    Updates both Style definition lines and style references in
    Dialogue/Comment event lines.
    """
    lines = text.splitlines()
    style_names: list[str] = []
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Style:"):
            parts = stripped.split(",", 1)
            name = parts[0].replace("Style:", "").strip()
            new_name = name + suffix
            style_names.append(name)
            out.append(f"Style: {new_name},{parts[1]}")
            continue
        out.append(line)

    if not style_names:
        return "\n".join(out)

    result_lines: list[str] = []
    for line in out:
        stripped = line.strip()
        if stripped.startswith("Dialogue:") or stripped.startswith("Comment:"):
            parts = stripped.split(",", 4)
            if len(parts) >= 5:
                style = parts[3].strip()
                if style in style_names:
                    parts[3] = style + suffix
                line = ",".join(parts)
        result_lines.append(line)

    return "\n".join(result_lines)


def strip_position_tags(text: str) -> str:
    r"""Remove \pos(...) and \move(...) from dialogue events only.

    Sign/title events keep their position tags because they are
    designed to match specific on-screen text placements.
    """
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Dialogue:") or stripped.startswith("Comment:"):
            parts = stripped.split(",", 4)
            if len(parts) >= 5:
                style = parts[3].strip()
                if _is_dialogue_style(style):
                    line = POS_TAG_RE.sub("", line)
        out.append(line)
    return "\n".join(out)


def _read_ass_text(path: Path) -> str:
    """Read an ASS file with encoding fallback."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_ass_sections(path: Path) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Parse an ASS file into (script_info, style_lines, events).

    script_info: lines from [Script Info] (including section header)
    style_lines: lines from [V4+ Styles] (including section header + Format line)
    events: list of (start_time, raw_line) for Dialogue/Comment lines
    """
    text = _read_ass_text(path)
    lines = text.splitlines()

    script_info: list[str] = []
    styles: list[str] = []
    events: list[tuple[str, str]] = []
    current: list[str] | None = None

    for line in lines:
        stripped = line.strip()
        if stripped == "[Script Info]":
            current = script_info
            current.append(stripped)
            continue
        if stripped == "[V4+ Styles]":
            current = styles
            current.append(stripped)
            continue
        if stripped == "[Events]":
            current = None
            continue
        if stripped.startswith("["):
            current = None
            continue

        if current is not None and stripped:
            if stripped.startswith(";"):
                continue
            current.append(stripped)

        if stripped.startswith("Dialogue:") or stripped.startswith("Comment:"):
            parts = stripped.split(",", 2)
            if len(parts) >= 3:
                events.append((parts[1].strip(), stripped))

    return script_info, styles, events


def parse_ass_events(path: Path) -> list[tuple[str, str]]:
    """Read an ASS file and return (start_time, raw_line) for every event."""
    _, _, events = parse_ass_sections(path)
    return events


SIGN_MODES = ("shift", "strip")


def build_ass_merged(
    script_info: list[str],
    eng_styles: list[str],
    por_styles: list[str],
    eng_events: list[tuple[str, str]],
    por_events: list[tuple[str, str]],
    sign_mode: str = "shift",
) -> str:
    """Build a merged ASS file combining styles and events from both sources.

    Uses eng script_info as base.  Combines English styles + Portuguese
    (already _BR-renamed) styles.  Events are interleaved by start time.

    sign_mode controls how Portuguese sign/title events are handled:
      "shift" — dynamically shift \\pos() Y below the nearest English sign
      "strip" — remove \\pos()/\\move() so signs use default alignment
    """
    lines: list[str] = []

    for l in script_info:
        if l.startswith("Title:") or l.startswith("Collisions:"):
            continue
        lines.append(l)
    lines.insert(1, f"Title: {TRACK_TITLE}")
    lines.insert(2, "Language: pt")
    lines.append("")

    lines.append("[V4+ Styles]")
    fmt_line = None
    eng_style_lines: list[str] = []
    for l in eng_styles:
        if l.startswith("Format:"):
            fmt_line = l
        elif l.startswith("Style:"):
            eng_style_lines.append(l)

    por_style_lines: list[str] = []
    for l in por_styles:
        if l.startswith("Format:") and fmt_line is None:
            fmt_line = l
        elif l.startswith("Style:"):
            por_style_lines.append(l)

    eng_style_lines = adjust_eng_styles(eng_style_lines)

    play_res_y = _get_play_res_y(script_info)
    max_eng_bot_mv = _get_max_bottom_dialogue_margin(eng_style_lines)
    por_bot_mv = play_res_y - max_eng_bot_mv + POR_BOT_GAP

    max_eng_top_bottom = _get_max_top_dialogue_bottom(eng_style_lines)
    por_top_mv = max_eng_top_bottom + POR_BOT_GAP if max_eng_top_bottom > 0 else None

    por_style_lines = adjust_por_styles(
        por_style_lines,
        por_bot_margin=por_bot_mv,
        por_top_margin=por_top_mv,
    )

    if fmt_line:
        lines.append(fmt_line)
    for l in eng_style_lines:
        lines.append(l)
    for l in por_style_lines:
        lines.append(l)
    lines.append("")

    lines.append("[Events]")
    lines.append(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )

    all_style_lines = eng_style_lines + por_style_lines
    style_font_map = _build_style_font_map(all_style_lines)

    sign_offsets: dict[int, int] = {}
    if sign_mode == "shift":
        sign_offsets = _compute_sign_offsets(
            eng_events, por_events, style_font_map,
        )

    tagged: list[tuple[str, int, str]] = []
    for ts, raw in eng_events:
        tagged.append((ts, 0, _bump_event_layer(raw, ENG_LAYER_OFFSET)))
    for idx, (ts, raw) in enumerate(por_events):
        parts = raw.split(",", 4)
        if len(parts) >= 5:
            style = parts[3].strip()
            if not _is_dialogue_style(style):
                if sign_mode == "shift" and idx in sign_offsets:
                    raw = _offset_pos_y(raw, sign_offsets[idx])
                elif sign_mode == "strip":
                    raw = POS_TAG_RE.sub("", raw)
        tagged.append((ts, 1, raw))
    tagged.sort(key=lambda e: (e[0], e[1]))

    for _ts, _order, raw_line in tagged:
        lines.append(raw_line)

    return "\r\n".join(lines) + "\r\n"


ASS_BOTTOM_ALIGNMENTS = {"1", "2", "3"}
ASS_TOP_ALIGNMENTS = {"7", "8", "9"}
BOTTOM_TO_TOP_ALIGN = {"1": "7", "2": "8", "3": "9"}

# Portuguese subtitle appearance — yellow text, black outline
POR_PRIMARY_COLOUR = "&H0000FFFF"   # yellow (ASS is &HBBGGRR)
POR_OUTLINE_COLOUR = "&H00000000"   # black outline
POR_BACK_COLOUR = "&H80000000"      # semi-transparent shadow
POR_FONTSIZE_REDUCTION = 4          # px smaller than English
POR_BOT_GAP = 2                     # px gap between English bottom and Portuguese top
SIGN_PROXIMITY_PX = 60              # vertical proximity threshold for sign overlap detection

SIGN_STYLE_RE = re.compile(
    r"(?:sign|title|eyecatch|lyric|cred|show_|ep_|next_)",
    re.IGNORECASE,
)


def _is_dialogue_style(name: str) -> bool:
    """True if this style is spoken dialogue rather than on-screen sign/title."""
    clean = name.replace("_BR", "").strip()
    return SIGN_STYLE_RE.search(clean) is None


def _get_play_res_y(script_info: list[str]) -> int:
    """Extract PlayResY from script info lines (default 360)."""
    for line in script_info:
        if line.startswith("PlayResY:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return 360


def _get_max_bottom_dialogue_margin(style_lines: list[str]) -> int:
    """Find the largest MarginV among bottom-aligned dialogue styles."""
    max_mv = 0
    for line in style_lines:
        if not line.startswith("Style:"):
            continue
        parts = line.split(",")
        if len(parts) >= 22:
            name = parts[0].replace("Style:", "").strip()
            alignment = parts[18].strip()
            if alignment in ASS_BOTTOM_ALIGNMENTS and _is_dialogue_style(name):
                try:
                    max_mv = max(max_mv, int(parts[21].strip()))
                except ValueError:
                    pass
    return max_mv


def _get_max_top_dialogue_bottom(style_lines: list[str]) -> int:
    """Find the lowest bottom edge among top-aligned dialogue styles.

    For alignment 8 (top), the bottom edge is MarginV + FontSize.
    """
    max_bottom = 0
    for line in style_lines:
        if not line.startswith("Style:"):
            continue
        parts = line.split(",")
        if len(parts) >= 22:
            name = parts[0].replace("Style:", "").strip()
            alignment = parts[18].strip()
            if alignment in ASS_TOP_ALIGNMENTS and _is_dialogue_style(name):
                try:
                    mv = int(parts[21].strip())
                    fs = int(float(parts[2].strip()))
                    max_bottom = max(max_bottom, mv + fs)
                except ValueError:
                    pass
    return max_bottom


def adjust_eng_styles(style_lines: list[str], margin_offset: int = 20) -> list[str]:
    """Push English bottom-aligned *dialogue* styles up so they sit above Portuguese.

    Sign/title styles are left untouched.
    """
    out: list[str] = []
    for line in style_lines:
        if not line.startswith("Style:"):
            out.append(line)
            continue
        parts = line.split(",")
        if len(parts) >= 22:
            name = parts[0].replace("Style:", "").strip()
            alignment = parts[18].strip()
            if alignment in ASS_BOTTOM_ALIGNMENTS and _is_dialogue_style(name):
                try:
                    mv = int(parts[21].strip())
                    parts[21] = str(mv + margin_offset)
                except ValueError:
                    pass
        out.append(",".join(parts))
    return out


def adjust_por_styles(
    style_lines: list[str],
    por_bot_margin: int | None = None,
    por_top_margin: int | None = None,
) -> list[str]:
    """Make Portuguese *dialogue* styles visually distinct: smaller, yellow.

    Sign/title styles keep their original appearance since they're designed
    to match on-screen text.

    Bottom-aligned dialogue styles are flipped to top-alignment so their
    text grows *downward* from a fixed Y position (por_bot_margin) just
    below the English text.  This eliminates overlap regardless of line
    count.

    Top-aligned dialogue uses por_top_margin so it starts right below
    the English top text's bottom edge.
    """
    out: list[str] = []
    for line in style_lines:
        if not line.startswith("Style:"):
            out.append(line)
            continue
        parts = line.split(",")
        if len(parts) < 23:
            out.append(line)
            continue

        name = parts[0].replace("Style:", "").strip()
        is_dialogue = _is_dialogue_style(name)

        if is_dialogue:
            try:
                fs = float(parts[2].strip())
                parts[2] = str(max(10, int(fs - POR_FONTSIZE_REDUCTION)))
            except ValueError:
                pass

            parts[3] = POR_PRIMARY_COLOUR
            parts[5] = POR_OUTLINE_COLOUR
            parts[6] = POR_BACK_COLOUR

            alignment = parts[18].strip()
            if alignment in ASS_BOTTOM_ALIGNMENTS and por_bot_margin is not None:
                parts[18] = BOTTOM_TO_TOP_ALIGN[alignment]
                parts[21] = str(por_bot_margin)
            elif alignment in ASS_TOP_ALIGNMENTS and por_top_margin is not None:
                parts[21] = str(por_top_margin)

        out.append(",".join(parts))
    return out


def prepare_ass_for_merge(path: Path, is_portuguese: bool = False) -> Path:
    """Prepare an ASS file for merging.

    For Portuguese files: renames styles with _BR suffix.
    For all files: strips position tags to prevent overlap.
    Writes the modified file next to the original with a .prepared.ass suffix.
    Returns the path to the prepared file.
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    if is_portuguese:
        text = rename_ass_styles(text, "_BR")

    text = strip_position_tags(text)

    out_path = path.with_suffix(".prepared.ass")
    out_path.write_text(text, encoding="utf-8-sig")
    return out_path

# ---------------------------------------------------------------------------
# Saved templates config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.is_file():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(data: dict):
    CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_templates() -> dict[str, str]:
    return load_config().get("templates", {})


def save_templates(templates: dict[str, str]):
    cfg = load_config()
    cfg["templates"] = templates
    save_config(cfg)

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def _parse_dnd_paths(data: str) -> list[Path]:
    """Extract file paths from a tkdnd drop event data string.

    Tk wraps paths containing spaces in curly braces and separates
    multiple paths with spaces.
    """
    paths: list[Path] = []
    raw = data.strip()
    i = 0
    while i < len(raw):
        if raw[i] == "{":
            end = raw.index("}", i)
            paths.append(Path(raw[i + 1 : end]))
            i = end + 2
        elif raw[i] == " ":
            i += 1
        else:
            end = raw.find(" ", i)
            if end == -1:
                end = len(raw)
            paths.append(Path(raw[i:end]))
            i = end + 1
    return paths


class App(TkinterDnD.Tk if HAS_DND else tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch SRT to ASS Converter")
        self.minsize(860, 640)
        self.resizable(True, True)

        self.pairs: list[tuple[str, str, float]] = []
        self.unmatched_top: list[str] = []
        self.unmatched_bot: list[str] = []
        self.template_path: Path | None = None
        self.script_info: list[str] = []
        self.styles: list[str] = []

        self.top_file_paths: dict[str, Path] = {}
        self.bot_file_paths: dict[str, Path] = {}
        self.saved_templates: dict[str, str] = load_templates()

        self._build_ui()

    # ---- UI construction ----

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # -- MKV drop zone --
        mkv_frame = ttk.LabelFrame(self, text="MKV Import", padding=6)
        mkv_frame.pack(fill="x", **pad)
        self.mkv_label = ttk.Label(
            mkv_frame,
            text="Drag MKV file(s) here or click Browse to extract subtitles",
            anchor="center", padding=10, relief="groove",
        )
        self.mkv_label.pack(side="left", fill="x", expand=True)
        ttk.Button(mkv_frame, text="Browse MKV", command=self._browse_mkv).pack(side="left", padx=(6, 0))
        if HAS_DND:
            self.mkv_label.drop_target_register(DND_FILES)
            self.mkv_label.dnd_bind("<<Drop>>", self._on_drop_mkv)

        # -- Mode toggle --
        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill="x", **pad)
        self.mode_var = tk.StringVar(value="folder")
        ttk.Label(mode_frame, text="Mode:").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(mode_frame, text="Folder Mode", variable=self.mode_var,
                         value="folder", command=self._toggle_mode).pack(side="left", padx=4)
        ttk.Radiobutton(mode_frame, text="File Mode", variable=self.mode_var,
                         value="file", command=self._toggle_mode).pack(side="left", padx=4)
        ttk.Radiobutton(mode_frame, text="ASS Merge Mode", variable=self.mode_var,
                         value="ass_merge", command=self._toggle_mode).pack(side="left", padx=4)

        # -- Container that swaps between folder / file UI --
        self.input_container = ttk.Frame(self)
        self.input_container.pack(fill="both", **pad)

        # -- Template row with saved templates (hidden in ASS Merge Mode) --
        self.tpl_frame = ttk.LabelFrame(self, text="Template", padding=6)
        tpl_outer = self.tpl_frame

        tpl_top_row = ttk.Frame(tpl_outer)
        tpl_top_row.pack(fill="x")
        ttk.Label(tpl_top_row, text="Template .ass:").pack(side="left")
        self.tpl_var = tk.StringVar()
        tpl_entry = ttk.Entry(tpl_top_row, textvariable=self.tpl_var, width=50)
        tpl_entry.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(tpl_top_row, text="Browse", command=self._browse_tpl).pack(side="left")
        if HAS_DND:
            tpl_entry.drop_target_register(DND_FILES)
            tpl_entry.dnd_bind("<<Drop>>", self._on_drop_tpl)

        tpl_saved_row = ttk.Frame(tpl_outer)
        tpl_saved_row.pack(fill="x", pady=(4, 0))
        ttk.Label(tpl_saved_row, text="Saved:").pack(side="left")
        self.tpl_combo = ttk.Combobox(tpl_saved_row, state="readonly", width=30)
        self.tpl_combo.pack(side="left", padx=4)
        self.tpl_combo.bind("<<ComboboxSelected>>", self._on_template_selected)
        ttk.Button(tpl_saved_row, text="Save", command=self._save_template).pack(side="left", padx=2)
        ttk.Button(tpl_saved_row, text="Delete", command=self._delete_template).pack(side="left", padx=2)
        self._refresh_template_combo()

        # -- Auto-Match button --
        self._auto_match_btn = ttk.Button(self, text="Auto-Match", command=self._auto_match)
        self._auto_match_btn.pack(anchor="e", padx=12, pady=(4, 2))

        # -- Results table --
        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, **pad)

        cols = ("#", "top_srt", "bot_srt", "conf")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("#", text="#", anchor="center")
        self.tree.heading("top_srt", text="Top SRT", anchor="w")
        self.tree.heading("bot_srt", text="Bot SRT", anchor="w")
        self.tree.heading("conf", text="Conf.", anchor="center")
        self.tree.column("#", width=40, minwidth=30, stretch=False, anchor="center")
        self.tree.column("top_srt", width=280, minwidth=120)
        self.tree.column("bot_srt", width=280, minwidth=120)
        self.tree.column("conf", width=70, minwidth=50, stretch=False, anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # -- Bottom buttons --
        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", **pad)
        ttk.Button(btn_row, text="Edit Pair", command=self._edit_pair).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Remove Pair", command=self._remove_pair).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Convert All", command=self._convert_all).pack(side="right", padx=4)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=4).pack(
            fill="x", side="bottom", **pad
        )

        self._build_folder_ui()
        self._build_file_ui()
        self._build_ass_merge_ui()
        self._toggle_mode()

    # -- Folder mode UI --

    def _build_folder_ui(self):
        self.folder_frame = ttk.LabelFrame(self.input_container, text="Folders", padding=8)

        ttk.Label(self.folder_frame, text="Top Folder (English):").grid(row=0, column=0, sticky="w")
        self.top_dir_var = tk.StringVar()
        top_entry = ttk.Entry(self.folder_frame, textvariable=self.top_dir_var, width=55)
        top_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(self.folder_frame, text="Browse", command=self._browse_top_dir).grid(row=0, column=2)

        ttk.Label(self.folder_frame, text="Bot Folder (Other):").grid(row=1, column=0, sticky="w")
        self.bot_dir_var = tk.StringVar()
        bot_entry = ttk.Entry(self.folder_frame, textvariable=self.bot_dir_var, width=55)
        bot_entry.grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(self.folder_frame, text="Browse", command=self._browse_bot_dir).grid(row=1, column=2)

        self.folder_frame.columnconfigure(1, weight=1)

        if HAS_DND:
            for entry, var in [(top_entry, self.top_dir_var), (bot_entry, self.bot_dir_var)]:
                entry.drop_target_register(DND_FILES)
                entry.dnd_bind("<<Drop>>", self._make_folder_drop(var))

    # -- File mode UI --

    def _build_file_ui(self):
        self.file_frame = ttk.Frame(self.input_container)

        # Two side-by-side listboxes
        for col, (label_text, attr, add_cmd, clear_attr) in enumerate([
            ("Top SRT Files (English)", "top_listbox", "_browse_top_files", "top"),
            ("Bot SRT Files (Other)", "bot_listbox", "_browse_bot_files", "bot"),
        ]):
            col_frame = ttk.LabelFrame(self.file_frame, text=label_text, padding=4)
            col_frame.grid(row=0, column=col, sticky="nsew", padx=4)

            lb = tk.Listbox(col_frame, width=38, height=6, selectmode="extended")
            lb.pack(side="left", fill="both", expand=True)
            sb = ttk.Scrollbar(col_frame, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            sb.pack(side="left", fill="y")
            setattr(self, attr, lb)

            btn_fr = ttk.Frame(col_frame)
            btn_fr.pack(side="left", fill="y", padx=(4, 0))
            ttk.Button(btn_fr, text="+", width=3, command=getattr(self, add_cmd)).pack(pady=2)
            ttk.Button(btn_fr, text="-", width=3,
                       command=lambda a=attr: self._remove_selected_files(a)).pack(pady=2)

            if HAS_DND:
                lb.drop_target_register(DND_FILES)
                lb.dnd_bind("<<Drop>>", self._make_file_drop(attr))

        self.file_frame.columnconfigure(0, weight=1)
        self.file_frame.columnconfigure(1, weight=1)
        self.file_frame.rowconfigure(0, weight=1)

    # -- ASS merge mode UI --

    def _build_ass_merge_ui(self):
        self.ass_merge_frame = ttk.Frame(self.input_container)

        for col, (label_text, attr, add_cmd) in enumerate([
            ("English ASS Files", "ass_top_listbox", "_browse_ass_top_files"),
            ("Portuguese ASS Files", "ass_bot_listbox", "_browse_ass_bot_files"),
        ]):
            col_frame = ttk.LabelFrame(self.ass_merge_frame, text=label_text, padding=4)
            col_frame.grid(row=0, column=col, sticky="nsew", padx=4)

            lb = tk.Listbox(col_frame, width=38, height=6, selectmode="extended")
            lb.pack(side="left", fill="both", expand=True)
            sb = ttk.Scrollbar(col_frame, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            sb.pack(side="left", fill="y")
            setattr(self, attr, lb)

            btn_fr = ttk.Frame(col_frame)
            btn_fr.pack(side="left", fill="y", padx=(4, 0))
            ttk.Button(btn_fr, text="+", width=3, command=getattr(self, add_cmd)).pack(pady=2)
            ttk.Button(btn_fr, text="-", width=3,
                       command=lambda a=attr: self._remove_selected_files(a)).pack(pady=2)

            if HAS_DND:
                lb.drop_target_register(DND_FILES)
                lb.dnd_bind("<<Drop>>", self._make_ass_file_drop(attr))

        sign_frame = ttk.LabelFrame(self.ass_merge_frame, text="Sign / Title Handling", padding=4)
        sign_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(6, 0))
        self.sign_mode_var = tk.StringVar(value="shift")
        ttk.Radiobutton(
            sign_frame, text="Shift below English (keep original position, offset to avoid overlap)",
            variable=self.sign_mode_var, value="shift",
        ).pack(anchor="w")
        ttk.Radiobutton(
            sign_frame, text="Strip positioning (remove \\pos, let renderer auto-place)",
            variable=self.sign_mode_var, value="strip",
        ).pack(anchor="w")

        self.ass_merge_frame.columnconfigure(0, weight=1)
        self.ass_merge_frame.columnconfigure(1, weight=1)
        self.ass_merge_frame.rowconfigure(0, weight=1)

    def _toggle_mode(self):
        for child in self.input_container.winfo_children():
            child.pack_forget()
        mode = self.mode_var.get()
        if mode == "folder":
            self.folder_frame.pack(fill="x", expand=False)
            self.tpl_frame.pack(fill="x", padx=8, pady=4,
                                before=self._auto_match_btn)
        elif mode == "file":
            self.file_frame.pack(fill="both", expand=True)
            self.tpl_frame.pack(fill="x", padx=8, pady=4,
                                before=self._auto_match_btn)
        elif mode == "ass_merge":
            self.ass_merge_frame.pack(fill="both", expand=True)
            self.tpl_frame.pack_forget()

    # ---- Drag-and-drop helpers ----

    @staticmethod
    def _make_folder_drop(var: tk.StringVar):
        def handler(event):
            paths = _parse_dnd_paths(event.data)
            if paths:
                p = paths[0]
                var.set(str(p.parent if p.is_file() else p))
        return handler

    def _on_drop_tpl(self, event):
        paths = _parse_dnd_paths(event.data)
        for p in paths:
            if p.is_file():
                self.tpl_var.set(str(p))
                return

    # ---- MKV import ----

    def _browse_mkv(self):
        files = filedialog.askopenfilenames(
            title="Select MKV file(s)",
            filetypes=[("MKV files", "*.mkv"), ("All video", "*.mkv;*.mp4;*.avi;*.ts"), ("All files", "*.*")],
        )
        if files:
            self._process_mkvs([Path(f) for f in files])

    def _on_drop_mkv(self, event):
        try:
            paths = _parse_dnd_paths(event.data)
            mkvs = [p for p in paths if p.is_file() and p.suffix.lower() in (".mkv", ".mp4", ".avi", ".ts")]
            if mkvs:
                self._process_mkvs(mkvs)
            else:
                self.status_var.set("No video files found in drop")
        except Exception as e:
            messagebox.showerror("Drop error", f"Error processing drop:\n{e}")

    def _process_mkvs(self, mkv_paths: list[Path]):
        try:
            find_ffmpeg()
        except FileNotFoundError as e:
            messagebox.showerror("FFmpeg not found", str(e))
            return

        any_ass = False
        extracted = 0
        for mkv in mkv_paths:
            self.status_var.set(f"Scanning: {mkv.name}...")
            self.update_idletasks()
            try:
                tracks = probe_subtitles(mkv)
            except Exception as e:
                messagebox.showerror("Probe failed", f"{mkv.name}:\n{e}")
                continue

            if not tracks:
                messagebox.showwarning("No subtitles", f"No subtitle tracks found in:\n{mkv.name}")
                continue

            eng_tracks, por_tracks = auto_pick_tracks(tracks)

            eng_pick = self._resolve_track(mkv.name, "English (Top)", eng_tracks, tracks)
            if eng_pick is None:
                continue
            por_pick = self._resolve_track(mkv.name, "Portuguese (Bot)", por_tracks, tracks)
            if por_pick is None:
                continue

            stem = mkv.stem
            eng_codec = eng_pick.get("codec", "srt")
            por_codec = por_pick.get("codec", "srt")
            eng_ext = ".ass" if eng_codec in ("ass", "ssa") else ".srt"
            por_ext = ".ass" if por_codec in ("ass", "ssa") else ".srt"

            eng_out = mkv.parent / f"{stem}.en{eng_ext}"
            por_out = mkv.parent / f"{stem}.pt{por_ext}"

            try:
                self.status_var.set(f"Extracting English from {mkv.name}...")
                self.update_idletasks()
                eng_out = extract_subtitle(mkv, eng_pick["index"], eng_out, eng_codec)

                self.status_var.set(f"Extracting Portuguese from {mkv.name}...")
                self.update_idletasks()
                por_out = extract_subtitle(mkv, por_pick["index"], por_out, por_codec)
            except Exception as e:
                messagebox.showerror("Extraction failed", f"{mkv.name}:\n{e}")
                continue

            both_ass = eng_ext == ".ass" and por_ext == ".ass"
            if both_ass:
                any_ass = True
                self.status_var.set(f"Preparing ASS files for {mkv.name}...")
                self.update_idletasks()
                try:
                    eng_prepared = prepare_ass_for_merge(eng_out, is_portuguese=False)
                    por_prepared = prepare_ass_for_merge(por_out, is_portuguese=True)
                    eng_out = eng_prepared
                    por_out = por_prepared
                except Exception as e:
                    messagebox.showerror("Preparation failed", f"{mkv.name}:\n{e}")
                    continue

            eng_name = eng_out.name
            por_name = por_out.name
            if eng_name not in self.top_file_paths:
                self.top_file_paths[eng_name] = eng_out
            if por_name not in self.bot_file_paths:
                self.bot_file_paths[por_name] = por_out

            if both_ass:
                self.ass_top_listbox.insert("end", eng_name)
                self.ass_bot_listbox.insert("end", por_name)
            else:
                self.top_listbox.insert("end", eng_name)
                self.bot_listbox.insert("end", por_name)
            extracted += 1

        if any_ass:
            self.mode_var.set("ass_merge")
        else:
            self.mode_var.set("file")
        self._toggle_mode()

        self._update_file_count()
        if extracted:
            self.status_var.set(f"Extracted subtitles from {extracted} file(s). Click Auto-Match to pair.")

    def _resolve_track(self, mkv_name: str, label: str,
                       candidates: list[dict], all_tracks: list[dict]) -> dict | None:
        """Pick a single track. Auto-selects if exactly one candidate, otherwise prompts."""
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            return self._pick_track_dialog(
                mkv_name, label,
                f"No {label.split('(')[0].strip()} tracks auto-detected.\nChoose manually:",
                all_tracks,
            )
        return self._pick_track_dialog(
            mkv_name, label,
            f"Multiple {label.split('(')[0].strip()} tracks found.\nChoose one:",
            candidates,
        )

    def _pick_track_dialog(self, mkv_name: str, label: str,
                           message: str, tracks: list[dict]) -> dict | None:
        """Show a dialog for the user to pick a subtitle track."""
        dlg = tk.Toplevel(self)
        dlg.title(f"Select {label} - {mkv_name}")
        dlg.resizable(False, False)
        dlg.grab_set()

        ttk.Label(dlg, text=message, padding=8, wraplength=500).pack(anchor="w")

        cols = ("idx", "lang", "codec", "title")
        tree = ttk.Treeview(dlg, columns=cols, show="headings", height=min(len(tracks), 10),
                            selectmode="browse")
        tree.heading("idx", text="#")
        tree.heading("lang", text="Language")
        tree.heading("codec", text="Codec")
        tree.heading("title", text="Title")
        tree.column("idx", width=40, stretch=False)
        tree.column("lang", width=100)
        tree.column("codec", width=80)
        tree.column("title", width=280)
        tree.pack(padx=8, pady=4, fill="both")

        for t in tracks:
            tree.insert("", "end", iid=str(t["index"]),
                        values=(t["index"], t["language"], t["codec"], t["title"]))
        if tracks:
            tree.selection_set(str(tracks[0]["index"]))

        result: list[dict | None] = [None]

        def on_ok():
            sel = tree.selection()
            if sel:
                idx = int(sel[0])
                result[0] = next((t for t in tracks if t["index"] == idx), None)
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(fill="x", padx=8, pady=8)
        ttk.Button(bf, text="OK", command=on_ok).pack(side="right", padx=4)
        ttk.Button(bf, text="Cancel", command=on_cancel).pack(side="right", padx=4)

        dlg.wait_window()
        return result[0]

    # ---- Saved templates ----

    def _refresh_template_combo(self):
        names = sorted(self.saved_templates.keys())
        self.tpl_combo["values"] = names
        if names and not self.tpl_combo.get():
            self.tpl_combo.current(0)
            self._on_template_selected()

    def _on_template_selected(self, event=None):
        name = self.tpl_combo.get()
        path = self.saved_templates.get(name, "")
        if path:
            self.tpl_var.set(path)

    def _save_template(self):
        current_path = self.tpl_var.get().strip()
        if not current_path:
            messagebox.showwarning("No template", "Set a template .ass path first.")
            return
        name = simpledialog.askstring("Save Template", "Template name:", parent=self)
        if not name:
            return
        self.saved_templates[name] = current_path
        save_templates(self.saved_templates)
        self._refresh_template_combo()
        self.tpl_combo.set(name)
        self.status_var.set(f"Template '{name}' saved.")

    def _delete_template(self):
        name = self.tpl_combo.get()
        if not name or name not in self.saved_templates:
            messagebox.showinfo("Nothing selected", "Select a saved template to delete.")
            return
        self.saved_templates.pop(name)
        save_templates(self.saved_templates)
        self.tpl_combo.set("")
        self._refresh_template_combo()
        self.status_var.set(f"Template '{name}' deleted.")

    def _make_file_drop(self, listbox_attr: str):
        def handler(event):
            paths = _parse_dnd_paths(event.data)
            lb: tk.Listbox = getattr(self, listbox_attr)
            store = self.top_file_paths if "top" in listbox_attr else self.bot_file_paths
            for p in paths:
                if p.is_file() and p.suffix.lower() in (".srt", ".ass"):
                    name = p.name
                    if name not in store:
                        store[name] = p
                        lb.insert("end", name)
            self._update_file_count()
        return handler

    def _make_ass_file_drop(self, listbox_attr: str):
        def handler(event):
            paths = _parse_dnd_paths(event.data)
            lb: tk.Listbox = getattr(self, listbox_attr)
            store = self.top_file_paths if "top" in listbox_attr else self.bot_file_paths
            for p in paths:
                if p.is_file() and p.suffix.lower() in (".ass", ".ssa"):
                    name = p.name
                    if name not in store:
                        store[name] = p
                        lb.insert("end", name)
            self._update_file_count()
        return handler

    def _remove_selected_files(self, listbox_attr: str):
        lb: tk.Listbox = getattr(self, listbox_attr)
        store = self.top_file_paths if "top" in listbox_attr else self.bot_file_paths
        for idx in reversed(lb.curselection()):
            name = lb.get(idx)
            store.pop(name, None)
            lb.delete(idx)
        self._update_file_count()

    def _update_file_count(self):
        t = len(self.top_file_paths)
        b = len(self.bot_file_paths)
        self.status_var.set(f"{t} Top file(s), {b} Bot file(s)")

    # ---- Browse callbacks ----

    def _browse_top_dir(self):
        d = filedialog.askdirectory(title="Select Top (English) SRT folder")
        if d:
            self.top_dir_var.set(d)

    def _browse_bot_dir(self):
        d = filedialog.askdirectory(title="Select Bot (other language) SRT folder")
        if d:
            self.bot_dir_var.set(d)

    def _browse_tpl(self):
        f = filedialog.askopenfilename(
            title="Select template .ass file",
            filetypes=[("ASS files", "*.ass"), ("All files", "*.*")],
        )
        if f:
            self.tpl_var.set(f)

    def _browse_top_files(self):
        files = filedialog.askopenfilenames(
            title="Select Top (English) SRT files",
            filetypes=[("SRT files", "*.srt"), ("All files", "*.*")],
        )
        for f in files:
            p = Path(f)
            if p.name not in self.top_file_paths:
                self.top_file_paths[p.name] = p
                self.top_listbox.insert("end", p.name)
        self._update_file_count()

    def _browse_bot_files(self):
        files = filedialog.askopenfilenames(
            title="Select Bot (other language) SRT files",
            filetypes=[("SRT files", "*.srt"), ("All files", "*.*")],
        )
        for f in files:
            p = Path(f)
            if p.name not in self.bot_file_paths:
                self.bot_file_paths[p.name] = p
                self.bot_listbox.insert("end", p.name)
        self._update_file_count()

    def _browse_ass_top_files(self):
        files = filedialog.askopenfilenames(
            title="Select English ASS files",
            filetypes=[("ASS files", "*.ass;*.ssa"), ("All files", "*.*")],
        )
        for f in files:
            p = Path(f)
            if p.name not in self.top_file_paths:
                self.top_file_paths[p.name] = p
                self.ass_top_listbox.insert("end", p.name)
        self._update_file_count()

    def _browse_ass_bot_files(self):
        files = filedialog.askopenfilenames(
            title="Select Portuguese ASS files",
            filetypes=[("ASS files", "*.ass;*.ssa"), ("All files", "*.*")],
        )
        for f in files:
            p = Path(f)
            if p.name not in self.bot_file_paths:
                self.bot_file_paths[p.name] = p
                self.ass_bot_listbox.insert("end", p.name)
        self._update_file_count()

    # ---- Auto-match ----

    def _load_template(self) -> bool:
        tpl_path = self.tpl_var.get().strip()
        if not tpl_path:
            messagebox.showwarning("Missing template", "Please set the template .ass file.")
            return False
        self.template_path = Path(tpl_path)
        if not self.template_path.exists():
            messagebox.showerror("Not found", f"Template not found:\n{self.template_path}")
            return False
        try:
            self.script_info, self.styles = parse_ass_template(self.template_path)
        except Exception as e:
            messagebox.showerror("Template error", f"Failed to parse template:\n{e}")
            return False
        if not self.styles:
            messagebox.showerror("Template error", "No [V4+ Styles] section found in template.")
            return False
        return True

    def _auto_match(self):
        mode = self.mode_var.get()
        if mode != "ass_merge":
            if not self._load_template():
                return

        if mode == "folder":
            self._auto_match_folder()
        elif mode in ("file", "ass_merge"):
            self._auto_match_files()

    def _auto_match_folder(self):
        top_path = self.top_dir_var.get().strip()
        bot_path = self.bot_dir_var.get().strip()
        if not top_path or not bot_path:
            messagebox.showwarning("Missing paths", "Please set both folder paths.")
            return

        top_dir = Path(top_path)
        bot_dir = Path(bot_path)
        for label, p in [("Top folder", top_dir), ("Bot folder", bot_dir)]:
            if not p.exists():
                messagebox.showerror("Not found", f"{label} does not exist:\n{p}")
                return

        top_srts = sorted(f.name for f in top_dir.glob("*.srt"))
        bot_srts = sorted(f.name for f in bot_dir.glob("*.srt"))
        if not top_srts:
            messagebox.showwarning("No files", f"No .srt files in Top folder:\n{top_dir}")
            return
        if not bot_srts:
            messagebox.showwarning("No files", f"No .srt files in Bot folder:\n{bot_dir}")
            return

        self.top_file_paths = {f: top_dir / f for f in top_srts}
        self.bot_file_paths = {f: bot_dir / f for f in bot_srts}

        self.pairs, self.unmatched_top, self.unmatched_bot = match_files(top_srts, bot_srts)
        self._refresh_table()
        self.status_var.set(f"Matched {len(self.pairs)} pair(s), "
                            f"{len(self.unmatched_top) + len(self.unmatched_bot)} unmatched")

    def _auto_match_files(self):
        top_names = sorted(self.top_file_paths.keys())
        bot_names = sorted(self.bot_file_paths.keys())
        if not top_names:
            messagebox.showwarning("No files", "Add Top SRT files first.")
            return
        if not bot_names:
            messagebox.showwarning("No files", "Add Bot SRT files first.")
            return

        self.pairs, self.unmatched_top, self.unmatched_bot = match_files(top_names, bot_names)
        self._refresh_table()
        self.status_var.set(f"Matched {len(self.pairs)} pair(s), "
                            f"{len(self.unmatched_top) + len(self.unmatched_bot)} unmatched")

    # ---- Table refresh ----

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for i, (tf, bf, score) in enumerate(self.pairs, 1):
            tag = "low" if score < 0.8 else ""
            conf_text = f"{score:.0%}" + (" *" if score < 0.8 else "")
            self.tree.insert("", "end", iid=str(i), values=(i, tf, bf, conf_text), tags=(tag,))
        self.tree.tag_configure("low", foreground="#cc6600")

        if self.unmatched_top or self.unmatched_bot:
            self.tree.insert("", "end", iid="sep", values=("", "--- Unmatched ---", "", ""))
            for j, f in enumerate(self.unmatched_top):
                self.tree.insert("", "end", iid=f"ut_{j}", values=("", f, "(no match - Top)", ""))
            for j, f in enumerate(self.unmatched_bot):
                self.tree.insert("", "end", iid=f"ub_{j}", values=("", "(no match - Bot)", f, ""))

    # ---- Edit pair ----

    def _edit_pair(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a paired row first.")
            return
        iid = sel[0]
        try:
            idx = int(iid) - 1
        except ValueError:
            messagebox.showinfo("Invalid", "Select a paired row, not an unmatched entry.")
            return
        if idx < 0 or idx >= len(self.pairs):
            return

        tf, old_bf, _ = self.pairs[idx]
        available = sorted(set(self.unmatched_bot + [old_bf]))
        if not available:
            messagebox.showinfo("No files", "No Bot files available to choose from.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Edit Pair")
        dlg.resizable(False, False)
        dlg.grab_set()

        ttk.Label(dlg, text=f"Top:  {tf}", padding=8).pack(anchor="w")
        ttk.Label(dlg, text="Choose Bot file:", padding=(8, 0)).pack(anchor="w")

        listbox = tk.Listbox(dlg, width=60, height=min(len(available), 15))
        listbox.pack(padx=8, pady=4, fill="both", expand=True)
        for bf in available:
            listbox.insert("end", bf)
        try:
            listbox.selection_set(available.index(old_bf))
            listbox.see(available.index(old_bf))
        except ValueError:
            pass

        def on_ok():
            cs = listbox.curselection()
            if not cs:
                return
            new_bf = available[cs[0]]
            if old_bf != new_bf:
                if old_bf not in self.unmatched_bot:
                    self.unmatched_bot.append(old_bf)
                if new_bf in self.unmatched_bot:
                    self.unmatched_bot.remove(new_bf)
            self.pairs[idx] = (tf, new_bf, 1.0)
            self._refresh_table()
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill="x", padx=8, pady=8)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side="right", padx=4)

    # ---- Remove pair ----

    def _remove_pair(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a paired row first.")
            return
        iid = sel[0]
        try:
            idx = int(iid) - 1
        except ValueError:
            messagebox.showinfo("Invalid", "Select a paired row, not an unmatched entry.")
            return
        if idx < 0 or idx >= len(self.pairs):
            return

        removed = self.pairs.pop(idx)
        self.unmatched_top.append(removed[0])
        self.unmatched_bot.append(removed[1])
        self._refresh_table()
        self.status_var.set(f"Removed pair. {len(self.pairs)} pair(s) remaining.")

    # ---- Convert ----

    def _convert_all(self):
        if not self.pairs:
            messagebox.showwarning("Nothing to convert", "No pairs to convert. Run Auto-Match first.")
            return

        converted = 0
        errors = []
        out_dir: Path | None = None

        for tf, bf, _ in self.pairs:
            top_path = self.top_file_paths.get(tf)
            bot_path = self.bot_file_paths.get(bf)
            if not top_path or not bot_path:
                errors.append(f"{tf}: path lookup failed")
                continue

            out_name_str = output_name(tf)
            out_path = top_path.parent / out_name_str
            out_dir = top_path.parent

            is_ass_merge = (top_path.suffix.lower() in (".ass", ".ssa")
                            and bot_path.suffix.lower() in (".ass", ".ssa"))

            try:
                if is_ass_merge:
                    eng_info, eng_styles, eng_events = parse_ass_sections(top_path)
                    por_info, por_styles, por_events = parse_ass_sections(bot_path)
                    if not eng_events:
                        errors.append(f"{tf}: no events parsed (English)")
                        continue
                    if not por_events:
                        errors.append(f"{bf}: no events parsed (Portuguese)")
                        continue
                    merge_info = eng_info if eng_info else self.script_info
                    s_mode = getattr(self, "sign_mode_var", None)
                    ass_content = build_ass_merged(
                        merge_info, eng_styles, por_styles,
                        eng_events, por_events,
                        sign_mode=s_mode.get() if s_mode else "shift",
                    )
                else:
                    top_cues = parse_srt(top_path)
                    bot_cues = parse_srt(bot_path)
                    if not top_cues:
                        errors.append(f"{tf}: no cues parsed (Top)")
                        continue
                    if not bot_cues:
                        errors.append(f"{bf}: no cues parsed (Bot)")
                        continue
                    ass_content = build_ass(
                        self.script_info, self.styles, top_cues, bot_cues
                    )

                out_path.write_text(ass_content, encoding="utf-8-sig")
                converted += 1
            except Exception as e:
                errors.append(f"{tf}: {e}")

        msg = f"Converted {converted}/{len(self.pairs)} pair(s)."
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        where = out_dir or "output folder"
        self.status_var.set(f"Done! {converted} file(s) written to {where}")
        messagebox.showinfo("Conversion complete", msg)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
