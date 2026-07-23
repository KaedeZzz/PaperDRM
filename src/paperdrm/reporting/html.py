"""Canonical bilingual HTML rendering for stored V2 results."""

from __future__ import annotations

import base64
from html import escape
from math import isfinite
from typing import Any


def _fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    number = float(value)
    return f"{number:.{digits}f}" if isfinite(number) else "—"


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _warning_text(values: dict[str, Any]) -> str:
    warnings = values.get("confidence_warnings")
    if not isinstance(warnings, (list, tuple)) or not warnings:
        return "none"
    return ", ".join(_text(value) for value in warnings)


def _image_html(overlay_png: bytes | None, *, alt: str) -> str:
    if overlay_png is None:
        return ""
    encoded = base64.b64encode(overlay_png).decode("ascii")
    return (
        '<figure><img src="data:image/png;base64,'
        f'{encoded}" alt="{_text(alt)}"></figure>'
    )


def _styles() -> str:
    return """
body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#202124}
h1,h2{color:#17324d} .policy{padding:12px;background:#eef4f8;border-left:4px solid #3f6f91}
table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border-bottom:1px solid #ddd;padding:9px;text-align:left}
img{max-width:100%;height:auto} .experimental{font-style:italic;color:#5f6368} footer{margin-top:30px;color:#5f6368;font-size:.9rem}
""".strip()


def _english(values: dict[str, Any], overlay_png: bytes | None) -> str:
    policy = " · ".join(
        _text(values.get(key, "unknown"))
        for key in (
            "confidence_policy_version",
            "confidence_disposition",
            "confidence_reason",
        )
    )
    warning = _warning_text(values)
    period_warning = values.get("period_warning")
    period_note = (
        f"<p><strong>Period warning:</strong> {_text(period_warning)}</p>"
        if period_warning
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>PaperDRM V2 report — {_text(values.get('serial', 'unknown'))}</title><style>{_styles()}</style></head>
<body><h1>Paper laid-line analysis</h1>
<p>Dataset: <strong>{_text(values.get('serial', 'unknown'))}</strong> · Report date: {_text(values.get('today', ''))}</p>
<p class="policy"><strong>Policy {policy}</strong><br>Detection assessment: {_text(values.get('detect_confidence_en', 'Unknown'))}; warnings: {warning}</p>
{period_note}
<h2>Headline measurement</h2><table>
<tr><th>Global spectral period</th><td>{_fmt(values.get('period_mm'), 3)} mm</td></tr>
<tr><th>Global density</th><td>{_fmt(values.get('lines_per_cm'))} lines/cm</td></tr>
<tr><th>Historical context</th><td>{_text(values.get('context_en', 'not available'))}</td></tr>
</table>
<h2>Diagnostic evidence</h2><table>
<tr><th>Local median gap</th><td>{_fmt(values.get('local_median_mm'), 3)} mm (IQR {_fmt(values.get('local_iqr_lo_mm'), 3)}–{_fmt(values.get('local_iqr_hi_mm'), 3)} mm)</td></tr>
<tr><th>Local/global difference</th><td>{_fmt(values.get('local_median_error_pct'))}% across {int(values.get('n_gaps') or 0)} gaps</td></tr>
<tr><th>Self-contrast z</th><td>{_fmt(values.get('z'))}</td></tr>
<tr><th>Fit R² / frequency concentration</th><td>{_fmt(values.get('r2_k4'), 3)} / {_fmt(values.get('fc'), 3)}</td></tr>
<tr><th>Split-half stability</th><td>{_text(values.get('stability_en', 'Unavailable'))}</td></tr>
</table>
<h2>Wire-width estimate</h2><p class="experimental">Experimental: median FWHM {_fmt(values.get('fwhm_mm_median'), 3)} mm from {int(values.get('seg_valid') or 0)} valid segments.</p>
{_image_html(overlay_png, alt='Detected laid-line overlay')}
<footer>Technical source: {_text(values.get('technical_location', 'stored V2 run'))}. This report presents stored V2 evidence and policy; it does not reclassify confidence.</footer>
</body></html>"""


def _chinese(values: dict[str, Any], overlay_png: bytes | None) -> str:
    policy = " · ".join(
        _text(values.get(key, "unknown"))
        for key in (
            "confidence_policy_version",
            "confidence_disposition",
            "confidence_reason",
        )
    )
    warning = _warning_text(values)
    period_warning = values.get("period_warning")
    period_note = (
        f"<p><strong>周期警告：</strong>{_text(period_warning)}</p>"
        if period_warning
        else ""
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>PaperDRM V2 报告 — {_text(values.get('serial', 'unknown'))}</title><style>{_styles()}</style></head>
<body><h1>纸张帘纹分析</h1>
<p>数据集：<strong>{_text(values.get('serial', 'unknown'))}</strong> · 报告日期：{_text(values.get('today', ''))}</p>
<p class="policy"><strong>策略 {policy}</strong><br>检测判断：{_text(values.get('detect_confidence_zh', '未知'))}；警告：{warning}</p>
{period_note}
<h2>核心测量</h2><table>
<tr><th>全局频谱周期</th><td>{_fmt(values.get('period_mm'), 3)} mm</td></tr>
<tr><th>全局密度</th><td>{_fmt(values.get('lines_per_cm'))} 条/cm</td></tr>
<tr><th>历史参照</th><td>{_text(values.get('context_zh', '暂无数据'))}</td></tr>
</table>
<h2>诊断证据</h2><table>
<tr><th>局部间距中位数</th><td>{_fmt(values.get('local_median_mm'), 3)} mm（IQR {_fmt(values.get('local_iqr_lo_mm'), 3)}–{_fmt(values.get('local_iqr_hi_mm'), 3)} mm）</td></tr>
<tr><th>局部/全局差异</th><td>{_fmt(values.get('local_median_error_pct'))}%（{int(values.get('n_gaps') or 0)} 个间距）</td></tr>
<tr><th>自对比 z 值</th><td>{_fmt(values.get('z'))}</td></tr>
<tr><th>拟合 R² / 频率集中度</th><td>{_fmt(values.get('r2_k4'), 3)} / {_fmt(values.get('fc'), 3)}</td></tr>
<tr><th>分半稳定性</th><td>{_text(values.get('stability_zh', '不可用'))}</td></tr>
</table>
<h2>线宽估计</h2><p class="experimental">实验性指标：FWHM 中位数 {_fmt(values.get('fwhm_mm_median'), 3)} mm，来自 {int(values.get('seg_valid') or 0)} 个有效分段。</p>
{_image_html(overlay_png, alt='帘纹检测叠加图')}
<footer>技术来源：{_text(values.get('technical_location', 'stored V2 run'))}。本报告展示已存储的 V2 证据与策略，不在展示层重新判级。</footer>
</body></html>"""


def render_bilingual_reports(
    values: dict[str, Any],
    overlay_png: bytes | None = None,
) -> tuple[str, str]:
    """Render English and Chinese reports from canonical V2 report values."""

    return _english(values, overlay_png), _chinese(values, overlay_png)
