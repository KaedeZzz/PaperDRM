"""
Generate plain-language analysis reports (English + Chinese) for a PaperDRM dataset.

Usage:
    python scripts/generate_report.py --serial 10
    python scripts/generate_report.py --serial 10 --results-dir results

Reads JSON results from results/<serial>/ and writes:
    results/<serial>/report_en.html
    results/<serial>/report_zh.html

The HTML files are self-contained (overlay image is base64-embedded) and can be
printed to PDF from any browser with Ctrl+P → Save as PDF.
"""

import argparse
import base64
import json
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict:
    """Load all JSON result files from results/<serial>/."""
    def _load(name):
        p = results_dir / name
        return json.loads(p.read_text()) if p.exists() else {}

    return {
        "interval":    _load("interval_distribution.json"),
        "wire_width":  _load("wire_width_stats.json"),
        "split_half":  _load("split_half_stability.json"),
        "self_contrast": _load("self_contrast.json"),
        "fit_quality": _load("fit_quality.json"),
    }


def load_image_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------------------
# Interpret raw numbers into human-readable summaries
# ---------------------------------------------------------------------------

def interpret(data: dict, serial: str) -> dict:
    iv = data["interval"]
    ww = data["wire_width"]
    sh = data["split_half"]
    sc = data["self_contrast"]
    fq = data["fit_quality"]

    # Core measurements
    physical          = iv.get("physical") or {}
    cm_per_px         = physical.get("cm_per_px", float("nan"))
    spectral_interval_cm = physical.get("spectral_interval_cm")
    if spectral_interval_cm is None:
        period_px = iv.get("period_px_used", float("nan"))
        spectral_interval_cm = period_px * cm_per_px
    period_mm         = spectral_interval_cm * 10
    lines_per_cm      = physical.get(
        "spectral_lines_per_cm",
        1.0 / spectral_interval_cm if spectral_interval_cm > 0 else float("nan"),
    )
    lines_per_cm_med  = physical.get("lines_per_cm_median", float("nan"))
    local_median_mm   = physical.get("median_interval_cm", float("nan")) * 10
    gap_iqr_cm        = physical.get("gap_iqr_cm")
    if gap_iqr_cm is None:
        gap_iqr_px = iv.get("px", {}).get("iqr", [float("nan"), float("nan")])
        gap_iqr_cm = [gap_iqr_px[0] * cm_per_px, gap_iqr_px[1] * cm_per_px]
    local_iqr_lo_mm   = gap_iqr_cm[0] * 10
    local_iqr_hi_mm   = gap_iqr_cm[1] * 10
    local_median_error_pct = physical.get(
        "gap_median_relative_error_vs_spectral",
        abs(local_median_mm - period_mm) / period_mm if period_mm > 0 else float("nan"),
    ) * 100
    n_peaks           = iv.get("n_peaks", 0)
    n_gaps            = iv.get("n_gaps", 0)

    fwhm_mm_median    = ww.get("physical", {}).get("fwhm_mm", {}).get("median", float("nan"))
    fwhm_mm_ci        = ww.get("physical", {}).get("fwhm_mm", {}).get("ci_t", [float("nan"), float("nan")])
    n_segments        = ww.get("n_segments", 0)
    seg_valid         = ww.get("aggregate", {}).get("fwhm_px", {}).get("n_valid", 0)

    n_phi             = sh.get("n_images", 0)
    n_splits          = sh.get("n_splits", 0)
    diff_std          = sh.get("period_diff_std", float("nan"))
    agree_1px         = sh.get("agree_rate_within_1px", float("nan"))
    agree_05px        = sh.get("agree_rate_within_0p5px", float("nan"))

    z                 = sc.get("contrast_z", float("nan"))
    contrast_rel      = sc.get("contrast_rel", float("nan"))
    n_lines           = sc.get("n_lines", 0)

    r2_k4             = fq.get("r2_with_harmonics", float("nan"))
    fc                = fq.get("frequency_concentration", float("nan"))
    period_at_boundary = bool(fq.get("period_at_search_boundary", False))
    period_warning     = fq.get("period_warning")

    # Reliability labels
    if period_at_boundary:
        detect_confidence_en = "Search boundary hit"
        detect_confidence_zh = "命中搜索边界"
    elif z <= -2.0:
        detect_confidence_en = "Contradictory polarity"
        detect_confidence_zh = "极性矛盾"
    elif z >= 3.0:
        detect_confidence_en = "High"
        detect_confidence_zh = "高"
    elif z >= 2.0:
        detect_confidence_en = "Moderate"
        detect_confidence_zh = "中"
    else:
        detect_confidence_en = "Low"
        detect_confidence_zh = "低"

    if diff_std == 0.0:
        stability_en = "Perfect (no variation across all splits)"
        stability_zh = "完全一致（所有分组结果相同）"
    elif diff_std < 0.5:
        stability_en = f"Excellent (±{diff_std:.3f} px)"
        stability_zh = f"优秀（±{diff_std:.3f} 像素）"
    elif diff_std < 1.5:
        stability_en = f"Good (±{diff_std:.3f} px)"
        stability_zh = f"良好（±{diff_std:.3f} 像素）"
    else:
        stability_en = f"Fair (±{diff_std:.3f} px)"
        stability_zh = f"一般（±{diff_std:.3f} 像素）"

    # Historical context
    if lines_per_cm < 8:
        context_en = "below the typical historical range (8–14 lines/cm)"
        context_zh = "低于历史常见范围（8–14 条/cm）"
    elif lines_per_cm <= 14:
        context_en = "within the typical historical range (8–14 lines/cm)"
        context_zh = "处于历史常见范围（8–14 条/cm）之内"
    else:
        context_en = "above the typical historical range (8–14 lines/cm)"
        context_zh = "高于历史常见范围（8–14 条/cm）"

    return dict(
        serial=serial,
        today=date.today().strftime("%Y-%m-%d"),
        lines_per_cm=lines_per_cm,
        lines_per_cm_med=lines_per_cm_med,
        period_mm=period_mm,
        local_median_mm=local_median_mm,
        local_iqr_lo_mm=local_iqr_lo_mm,
        local_iqr_hi_mm=local_iqr_hi_mm,
        local_median_error_pct=local_median_error_pct,
        n_gaps=n_gaps,
        n_peaks=n_peaks,
        fwhm_mm_median=fwhm_mm_median,
        fwhm_mm_ci_lo=fwhm_mm_ci[0],
        fwhm_mm_ci_hi=fwhm_mm_ci[1],
        n_segments=n_segments,
        seg_valid=seg_valid,
        n_phi=n_phi,
        n_splits=n_splits,
        diff_std=diff_std,
        agree_1px=agree_1px * 100,
        agree_05px=agree_05px * 100,
        z=z,
        contrast_rel=contrast_rel,
        n_lines=n_lines,
        r2_k4=r2_k4,
        fc=fc,
        period_at_boundary=period_at_boundary,
        period_warning=period_warning,
        detect_confidence_en=detect_confidence_en,
        detect_confidence_zh=detect_confidence_zh,
        stability_en=stability_en,
        stability_zh=stability_zh,
        context_en=context_en,
        context_zh=context_zh,
    )


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px;
       color: #222; background: #fff; max-width: 860px; margin: 40px auto;
       padding: 0 32px; }
h1 { font-size: 22px; margin-bottom: 4px; color: #1a3a5c; }
h2 { font-size: 16px; margin: 28px 0 10px; color: #1a3a5c;
     border-bottom: 2px solid #d0dce8; padding-bottom: 4px; }
h3 { font-size: 13px; margin: 16px 0 6px; color: #444; }
.subtitle { font-size: 13px; color: #666; margin-bottom: 24px; }
.highlight { background: #f0f5fb; border-left: 4px solid #2e6da4;
             padding: 14px 18px; margin: 16px 0; border-radius: 3px; }
.highlight .val { font-size: 26px; font-weight: bold; color: #1a3a5c; }
.highlight .label { font-size: 12px; color: #555; margin-top: 2px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
th { background: #2e6da4; color: #fff; padding: 7px 12px; text-align: left; }
td { padding: 6px 12px; border-bottom: 1px solid #e0e8f0; }
tr:nth-child(even) td { background: #f8fafc; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 12px; font-weight: bold; }
.badge-green { background: #d4edda; color: #155724; }
.badge-blue  { background: #d0e4f5; color: #0c4a8c; }
.badge-orange{ background: #fff3cd; color: #856404; }
.badge-red   { background: #f8d7da; color: #721c24; }
.warning-card { background: #f8d7da; color: #721c24; border-left: 4px solid #b02a37;
                padding: 12px 16px; margin: 12px 0; line-height: 1.5; }
img.overlay { width: 100%; max-width: 800px; border: 1px solid #ccc;
              border-radius: 4px; margin: 10px 0; }
.note { font-size: 12px; color: #666; margin-top: 6px; }
.section-intro { color: #444; line-height: 1.6; margin-bottom: 12px; }
.metric-card { background: #e8f2fc; border: 2px solid #2e6da4; border-radius: 6px;
               padding: 20px 26px; margin: 10px 0; }
.metric-label { font-size: 12px; color: #2e6da4; text-transform: uppercase;
                letter-spacing: 0.6px; font-weight: bold; margin-bottom: 4px; }
.metric-value { font-size: 42px; font-weight: bold; color: #1a3a5c; line-height: 1.1; }
.metric-unit  { font-size: 20px; font-weight: normal; color: #1a3a5c; }
.metric-ci    { font-size: 15px; color: #1a5c3a; font-weight: bold; margin-top: 6px;
                background: #d4edda; display: inline-block; padding: 3px 10px;
                border-radius: 4px; }
.metric-sub   { font-size: 12px; color: #555; margin-top: 6px; }
footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid #ddd;
         font-size: 11px; color: #999; }
@media print {
  body { max-width: 100%; padding: 0 16px; }
  .highlight { break-inside: avoid; }
}
"""


def _badge(label: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{label}</span>'


def _fmt(val, fmt=".2f", fallback="—") -> str:
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def build_html_en(v: dict, img_b64: str | None) -> str:
    img_tag = ""
    if img_b64:
        img_tag = f'<img class="overlay" src="data:image/png;base64,{img_b64}" alt="Overlay"/>'

    stability_badge = _badge(v["stability_en"], "green" if v["diff_std"] == 0.0 else "blue")
    if v["detect_confidence_en"] == "High":
        confidence_kind = "green"
    elif v["detect_confidence_en"] in {"Contradictory polarity", "Search boundary hit"}:
        confidence_kind = "red"
    else:
        confidence_kind = "orange"
    confidence_badge = _badge(v["detect_confidence_en"], confidence_kind)
    if v["period_at_boundary"]:
        spatial_interpretation = (
            "The spectral maximum is pinned to the configured period-search boundary. "
            "The spacing and density outputs are not validated until the range is corrected."
        )
    elif v["detect_confidence_en"] == "Contradictory polarity":
        spatial_interpretation = (
            "The grid aligns with the opposite intensity polarity from the configured "
            "wire model. Treat the detection as unconfirmed until polarity or phase is corrected."
        )
    else:
        spatial_interpretation = (
            "The predicted grid lines align with the expected brighter/darker columns "
            "in the actual image, supporting the detected grid."
        )
    boundary_note = ""
    if v["period_at_boundary"]:
        warning = v["period_warning"] or "Detected period is pinned to the search boundary."
        boundary_note = (
            '<div class="warning-card"><b>Invalid period search range.</b> '
            f"{warning}</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><title>PaperDRM Report — Dataset {v['serial']}</title>
<style>{_CSS}</style></head>
<body>

<h1>PaperDRM Laid-Line Analysis Report</h1>
<div class="subtitle">Dataset: <b>{v['serial']}</b> &nbsp;|&nbsp; Generated: {v['today']} &nbsp;|&nbsp; PaperDRM automated analysis</div>

<h2>Background</h2>
<p class="section-intro">
Historical paper was made by pouring pulp onto a wire mould. The mould consisted of closely
spaced parallel wires (called <em>laid lines</em>) held together by wider crossing wires
(<em>chain lines</em>). As the paper dried, the shadow of these wires was pressed into its
structure and remains faintly visible to this day.
</p>
<p class="section-intro">
By measuring how tightly the laid lines are spaced we can characterise the paper mould and,
in some cases, identify the paper-mill or date of manufacture. This report presents
measurements obtained by photographing the manuscript under many different raking-light
directions and combining the images computationally.
</p>

<h2>Key Findings</h2>
{boundary_note}

<div class="grid-2">
  <div class="metric-card">
    <div class="metric-label">Global Laid Line Spacing</div>
    <div class="metric-value">{_fmt(v['period_mm'], '.2f')}<span class="metric-unit"> mm</span></div>
    <div class="metric-ci">Local gaps: median {_fmt(v['local_median_mm'], '.2f')} mm
      &nbsp;|&nbsp; IQR {_fmt(v['local_iqr_lo_mm'], '.2f')} – {_fmt(v['local_iqr_hi_mm'], '.2f')} mm</div>
    <div class="metric-sub">Global spectral estimate = {_fmt(v['lines_per_cm'], '.1f')} lines/cm
      &nbsp;|&nbsp; local median differs by {_fmt(v['local_median_error_pct'], '.1f')}%
      &nbsp;|&nbsp; {v['context_en']}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Wire Shadow Width (FWHM)</div>
    <div class="metric-value">{_fmt(v['fwhm_mm_median'], '.3f')}<span class="metric-unit"> mm</span></div>
    <div class="metric-ci">95% CI &nbsp; {_fmt(v['fwhm_mm_ci_lo'], '.3f')} – {_fmt(v['fwhm_mm_ci_hi'], '.3f')} mm</div>
    <div class="metric-sub">Segment median &nbsp;|&nbsp; {v['seg_valid']}/{v['n_segments']} image segments valid</div>
  </div>
</div>

<div class="grid-2">
  <div class="highlight">
    <div class="val">{_fmt(v['lines_per_cm'], '.2f')} lines/cm</div>
    <div class="label">Laid-line density (mean)</div>
    <div class="note">Median: {_fmt(v['lines_per_cm_med'], '.2f')} lines/cm</div>
  </div>
  <div class="highlight">
    <div class="val">{v['n_peaks']}</div>
    <div class="label">Laid-line peaks identified in image</div>
    <div class="note">Used to measure the spacing distribution</div>
  </div>
</div>

<h2>Detection Reliability</h2>
<p class="section-intro">
Two independent checks assess whether the detected grid corresponds to the manuscript's
laid lines rather than noise or artefacts.
</p>

<table>
  <tr><th>Check</th><th>Result</th><th>Interpretation</th></tr>
  <tr>
    <td><b>Repeatability<br/><span class="note">(split-half test)</span></b></td>
    <td>{stability_badge}<br/>
        <span class="note">{_fmt(v['agree_05px'], '.0f')}% of {v['n_splits']} trials agree within ±0.5 px<br/>
        {_fmt(v['agree_1px'], '.0f')}% agree within ±1 px | based on {v['n_phi']} lighting directions</span></td>
    <td>The period measurement is stable regardless of which half of the
        lighting directions are used. This supports numerical repeatability,
        but does not by itself establish that the selected period is correct.</td>
  </tr>
  <tr>
    <td><b>Spatial consistency<br/><span class="note">(grid-vs-image check)</span></b></td>
    <td>{confidence_badge}<br/>
        <span class="note">z = {_fmt(v['z'], '+.2f')} (relative contrast = {_fmt(v['contrast_rel'] * 100, '+.1f')}%,
        across {v['n_lines']} lines)</span></td>
    <td>{spatial_interpretation}</td>
  </tr>
</table>

<h2>Visual Result</h2>
<p class="section-intro">
The image below shows the detected laid lines overlaid on the manuscript photograph
(raking light, phi = 0°). Blue bands mark each detected wire shadow; band width
reflects the measured wire thickness.
</p>
{img_tag if img_tag else '<p class="note">(overlay image not found)</p>'}

<h2>Measurement Setup (Summary)</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Lighting directions used</td><td>{v['n_phi']} azimuthal angles</td></tr>
  <tr><td>Spectral fit quality (R²)</td><td>{_fmt(v['r2_k4'], '.3f')} (Fourier, 4 harmonics)</td></tr>
  <tr><td>Frequency concentration</td><td>{_fmt(v['fc'], '.3f')} (fraction of power near peak)</td></tr>
  <tr><td>Analysis method</td><td>Multi-phi radial FFT aggregation with polarity correction</td></tr>
</table>

<footer>
  Generated by PaperDRM &nbsp;|&nbsp; Dataset {v['serial']} &nbsp;|&nbsp; {v['today']}<br/>
  For technical details see the accompanying JSON files in results/{v['serial']}/.
</footer>

</body></html>"""


def build_html_zh(v: dict, img_b64: str | None) -> str:
    img_tag = ""
    if img_b64:
        img_tag = f'<img class="overlay" src="data:image/png;base64,{img_b64}" alt="叠加图"/>'

    stability_badge = _badge(v["stability_zh"], "green" if v["diff_std"] == 0.0 else "blue")
    if v["detect_confidence_zh"] == "高":
        confidence_kind = "green"
    elif v["detect_confidence_zh"] in {"极性矛盾", "命中搜索边界"}:
        confidence_kind = "red"
    else:
        confidence_kind = "orange"
    confidence_badge = _badge(v["detect_confidence_zh"], confidence_kind)
    if v["period_at_boundary"]:
        spatial_interpretation = (
            "谱峰位于配置的周期搜索边界。在修正搜索范围之前，"
            "间距和密度结果不能视为已验证。"
        )
    elif v["detect_confidence_zh"] == "极性矛盾":
        spatial_interpretation = (
            "网格与配置的线影明暗极性相反。在修正极性或相位之前，"
            "该检测结果不能视为已确认。"
        )
    else:
        spatial_interpretation = (
            "预测网格与实际图像中预期的明暗列对齐，支持该纹线检测结果。"
        )
    boundary_note = ""
    if v["period_at_boundary"]:
        boundary_note = (
            '<div class="warning-card"><b>周期搜索范围无效。</b>'
            "检测峰命中搜索边界；请修正 period_range_cm 后重新运行。</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"/><title>PaperDRM 分析报告 — 数据集 {v['serial']}</title>
<style>{_CSS}</style></head>
<body>

<h1>PaperDRM 帘纹线分析报告</h1>
<div class="subtitle">数据集：<b>{v['serial']}</b> &nbsp;|&nbsp; 生成日期：{v['today']} &nbsp;|&nbsp; PaperDRM 自动分析</div>

<h2>背景介绍</h2>
<p class="section-intro">
历史上的手工纸是将纸浆倒入竹帘或铁丝网模具制成的。模具由紧密排列的平行细丝（称为<em>帘纹线</em>，即 laid lines）
以及间距较大的横向粗丝（链线，chain lines）构成。纸张干燥时，这些丝线的印记被压入纸中，至今仍以细微的纹路形式保留。
</p>
<p class="section-intro">
通过测量帘纹线的间距，可以识别制纸模具的特征，进而在某些情况下推断造纸作坊或制作年代。
本报告通过在多个不同方向的掠射光下拍摄手稿图像，并对结果进行计算分析，给出定量测量结果。
</p>

<h2>主要发现</h2>
{boundary_note}

<div class="grid-2">
  <div class="metric-card">
    <div class="metric-label">全局帘纹线间距</div>
    <div class="metric-value">{_fmt(v['period_mm'], '.2f')}<span class="metric-unit"> mm</span></div>
    <div class="metric-ci">局部间隔：中位数 {_fmt(v['local_median_mm'], '.2f')} mm
      &nbsp;|&nbsp; IQR {_fmt(v['local_iqr_lo_mm'], '.2f')} – {_fmt(v['local_iqr_hi_mm'], '.2f')} mm</div>
    <div class="metric-sub">全局频谱估计 = {_fmt(v['lines_per_cm'], '.1f')} 条/cm
      &nbsp;|&nbsp; 局部中位数偏差 {_fmt(v['local_median_error_pct'], '.1f')}%
      &nbsp;|&nbsp; {v['context_zh']}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">线影宽度（FWHM）</div>
    <div class="metric-value">{_fmt(v['fwhm_mm_median'], '.3f')}<span class="metric-unit"> mm</span></div>
    <div class="metric-ci">95% 置信区间 &nbsp; {_fmt(v['fwhm_mm_ci_lo'], '.3f')} – {_fmt(v['fwhm_mm_ci_hi'], '.3f')} mm</div>
    <div class="metric-sub">分段中位值 &nbsp;|&nbsp; {v['seg_valid']}/{v['n_segments']} 个分段有效</div>
  </div>
</div>

<div class="grid-2">
  <div class="highlight">
    <div class="val">{_fmt(v['lines_per_cm'], '.2f')} 条/cm</div>
    <div class="label">帘纹线密度（均值）</div>
    <div class="note">中位值：{_fmt(v['lines_per_cm_med'], '.2f')} 条/cm</div>
  </div>
  <div class="highlight">
    <div class="val">{v['n_peaks']}</div>
    <div class="label">图像中识别出的帘纹线数量</div>
    <div class="note">用于统计间距分布</div>
  </div>
</div>

<h2>检测可靠性</h2>
<p class="section-intro">
以下两项独立检验用于评估所检测的纹线网格是否对应手稿中真实的帘纹线，而非噪声或伪影。
</p>

<table>
  <tr><th>验证方式</th><th>结果</th><th>说明</th></tr>
  <tr>
    <td><b>重复性检验<br/><span class="note">（split-half 测试）</span></b></td>
    <td>{stability_badge}<br/>
        <span class="note">{_fmt(v['agree_05px'], '.0f')}% 的 {v['n_splits']} 次随机试验结果差异 ≤ 0.5 像素<br/>
        {_fmt(v['agree_1px'], '.0f')}% 差异 ≤ 1 像素 | 基于 {v['n_phi']} 个光照方向</span></td>
    <td>无论随机选取哪一半光照方向进行分析，周期估计始终一致，
        说明结果并非由少数特殊角度主导；但重复性本身不能证明所选周期一定正确。</td>
  </tr>
  <tr>
    <td><b>空间一致性检验<br/><span class="note">（网格对图像验证）</span></b></td>
    <td>{confidence_badge}<br/>
        <span class="note">z = {_fmt(v['z'], '+.2f')}（相对对比度 {_fmt(v['contrast_rel'] * 100, '+.1f')}%，
        跨 {v['n_lines']} 条线统计）</span></td>
    <td>{spatial_interpretation}</td>
  </tr>
</table>

<h2>可视化结果</h2>
<p class="section-intro">
下图为帘纹线检测结果叠加于手稿照片（掠射光，phi = 0°）。蓝色色带标注每条检测到的帘纹线位置，
色带宽度反映测量所得的线影厚度。
</p>
{img_tag if img_tag else '<p class="note">（未找到叠加图像）</p>'}

<h2>测量说明（概要）</h2>
<table>
  <tr><th>参数</th><th>数值</th></tr>
  <tr><td>使用的光照方向数</td><td>{v['n_phi']} 个方位角</td></tr>
  <tr><td>谱拟合质量（R²）</td><td>{_fmt(v['r2_k4'], '.3f')}（Fourier，4 阶谐波）</td></tr>
  <tr><td>频率集中度</td><td>{_fmt(v['fc'], '.3f')}（峰值附近的功率占比）</td></tr>
  <tr><td>分析方法</td><td>多方向径向 FFT 聚合 + 极性自动校正</td></tr>
</table>

<footer>
  由 PaperDRM 自动生成 &nbsp;|&nbsp; 数据集 {v['serial']} &nbsp;|&nbsp; {v['today']}<br/>
  技术细节详见 results/{v['serial']}/ 目录下的 JSON 文件。
</footer>

</body></html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate plain-language HTML reports from PaperDRM results.")
    parser.add_argument("--serial", required=True, help="Data serial number (e.g. 10)")
    parser.add_argument("--results-dir", default="results", help="Root results directory (default: results)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) / args.serial
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    data = load_results(results_dir)
    v = interpret(data, args.serial)

    img_b64 = load_image_b64(results_dir / "laid_lines_overlay_bands.png")

    en_html = build_html_en(v, img_b64)
    zh_html = build_html_zh(v, img_b64)

    en_path = results_dir / "report_en.html"
    zh_path = results_dir / "report_zh.html"
    en_path.write_text(en_html, encoding="utf-8")
    zh_path.write_text(zh_html, encoding="utf-8")

    print(f"Reports written:")
    print(f"  English : {en_path}")
    print(f"  Chinese : {zh_path}")
    print()
    print(f"  Lines/cm  : {v['lines_per_cm']:.2f}  |  Spacing: {v['period_mm']:.2f} mm")
    print(f"  Wire FWHM : {v['fwhm_mm_median']:.3f} mm  |  Stability: {v['stability_en']}")
    print(f"  Confidence: {v['detect_confidence_en']}  (z = {v['z']:+.2f})")


if __name__ == "__main__":
    main()
