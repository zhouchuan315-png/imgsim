"""Self-contained HTML report.

Thumbnails are embedded as base64 so the output is a single file you can email or
drop into a ticket without worrying about relative paths breaking.
"""

from __future__ import annotations

import base64
import html
import io
import time
from pathlib import Path

from ._io import load_thumbnail
from ._types import Result

_UNIT = {"hash": "hamming", "embed": "cosine"}

_STYLE = """
*{box-sizing:border-box}
body{margin:0;padding:32px;background:#f6f7f9;color:#16181d;font:14px/1.6 -apple-system,
"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}
h1{font-size:24px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em}
.meta{color:#6b7280;margin-bottom:24px;font-size:13px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}
.stat{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 20px;min-width:132px}
.stat b{display:block;font-size:22px;font-weight:600;color:#2563eb;line-height:1.2}
.stat span{color:#6b7280;font-size:12px}
.group{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:18px;margin-bottom:14px}
.ghead{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.ghead strong{font-size:15px;font-weight:600}
.tag{background:#fef2f2;color:#dc2626;border-radius:6px;padding:2px 9px;font-size:12px;font-weight:600}
.tag.calm{background:#eff6ff;color:#2563eb}
.grid{display:flex;gap:12px;flex-wrap:wrap}
.item{width:150px;text-align:center}
.item img{width:150px;height:150px;object-fit:cover;border-radius:8px;border:1px solid #e5e7eb;
background:#fafafa;display:block}
.item .nm{font-size:11px;color:#4b5563;margin-top:5px;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}
.item .d{font-size:11px;color:#9ca3af;font-variant-numeric:tabular-nums}
.more{display:flex;align-items:center;justify-content:center;width:150px;height:150px;
border:1px dashed #d1d5db;border-radius:8px;color:#9ca3af;font-size:13px}
.empty{background:#fff;border:1px dashed #d1d5db;border-radius:12px;padding:56px;
text-align:center;color:#6b7280}
footer{margin-top:28px;color:#9ca3af;font-size:12px}
"""


def _thumbnail_data_uri(path: Path, size: int = 150) -> str:
    image = load_thumbnail(path, size)
    if image is None:
        return ""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def render_html(result: Result, max_items_per_group: int = 24, thumbnail_size: int = 150) -> str:
    """Build the report markup.

    Args:
        result: Output of :meth:`imgsim.ImageIndex.compare`.
        max_items_per_group: Show at most this many thumbnails per group; the rest
            collapse into a "+N more" tile. Keeps a 500-image duplicate cluster from
            producing a 20 MB file.
        thumbnail_size: Pixel size of embedded thumbnails.

    Returns:
        A complete HTML document as a string.
    """
    unit = _UNIT.get(result.kind, "distance")
    affected = sum(len(g) for g in result.groups)
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    # A 15-image corpus compares in well under a millisecond; "0.0s" reads like a bug.
    elapsed_text = f"{result.elapsed:.1f}s" if result.elapsed >= 0.05 else "&lt;0.1s"

    parts = [
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>imgsim report</title><style>", _STYLE, "</style></head><body>",
        "<h1>Image similarity report</h1>",
        f'<div class="meta">mode <b>{html.escape(result.kind)}</b> &middot; threshold '
        f"<b>{result.threshold:g}</b> &middot; {unit} distance &middot; generated {generated}"
        "</div>",
        '<div class="stats">',
        f'<div class="stat"><b>{result.n_images}</b><span>images indexed</span></div>',
        f'<div class="stat"><b>{len(result.groups)}</b><span>similar groups</span></div>',
        f'<div class="stat"><b>{affected}</b><span>images in groups</span></div>',
        f'<div class="stat"><b>{len(result.pairs)}</b><span>similar pairs</span></div>',
        f'<div class="stat"><b>{elapsed_text}</b><span>compare time</span></div>',
        "</div>",
    ]

    if result.failed:
        parts.append(
            f'<div class="meta">{len(result.failed)} file(s) could not be decoded '
            "and were skipped.</div>"
        )

    if not result.groups:
        parts.append(
            '<div class="empty">No similar images at this threshold.<br>'
            "Try raising <code>--threshold</code>.</div>"
        )

    for number, group in enumerate(result.groups, 1):
        parts.append(
            f'<div class="group"><div class="ghead"><strong>Group {number}</strong>'
            f'<span class="tag">{len(group)} images</span></div><div class="grid">'
        )
        for path, distance in zip(
            group.paths[:max_items_per_group], group.distances[:max_items_per_group]
        ):
            uri = _thumbnail_data_uri(path, thumbnail_size)
            name = html.escape(path.name)
            shown = f"{distance:.0f}" if result.kind == "hash" else f"{distance:.3f}"
            parts.append(
                f'<div class="item"><img loading="lazy" src="{uri}" alt="{name}" '
                f'title="{html.escape(str(path))}">'
                f'<div class="nm" title="{name}">{name}</div>'
                f'<div class="d">{unit} {shown}</div></div>'
            )
        remaining = len(group) - max_items_per_group
        if remaining > 0:
            parts.append(f'<div class="more">+{remaining} more</div>')
        parts.append("</div></div>")

    parts.append("<footer>Generated by imgsim &middot; MIT licensed</footer></body></html>")
    return "".join(parts)


def save_html(result: Result, path: str | Path, **kwargs) -> Path:
    """Render and write the report. Returns the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(result, **kwargs), encoding="utf-8")
    return target
