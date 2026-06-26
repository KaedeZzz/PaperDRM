# IIB Project Report — PaperDRM

**Deadline: 2026-06-01.**

## CUED hard constraints

- **Total ≤ 53 A4 pages** = 1 title + ≤2 technical abstract + ≤50 main body
- **12 pt body font, ~25 mm margins** all round
- Smaller font allowed for bibliography / code appendix
- **Must include**: technical abstract (also archived separately, must contain name + College + project title), introduction, conclusions, risk-assessment retrospective appendix (≤1 page A4), environmental/social/ethical discussion
- **CUED departmental coversheet** must be filled, signed, dated, submitted alongside (download .docx from Department site)
- Code listings: select novel snippets only; pseudo-code where possible

## Build

```powershell
cd report
latexmk -pdf -bibtex main.tex
# or:
pdflatex main && biber main && pdflatex main && pdflatex main
```

(Swap to `xelatex` if the official Cambridge IIB template requires it.)

## Layout

```
report/
├── main.tex                      ← entry point, 12pt, 25mm margins
├── refs.bib                      ← biber bibliography
├── figures/                      ← LaTeX-only figures
│                                   (existing results/ is on graphicspath too)
├── figure_list.md                ← which figures go where + key data tables
└── chapters/
    ├── 00_titlepage.tex          (1 page; fills College/Supervisor)
    ├── technical_abstract.tex    (≤2 pages, self-contained — write LAST)
    ├── 01_introduction.tex       (~5 pp; includes prior work)
    ├── 02_theory.tex             (~12 pp; CUED "Theory and design")
    ├── 03_apparatus.tex          (~4 pp; CUED "Apparatus and techniques")
    ├── 04_results.tex            (~13 pp)
    ├── 05_discussion.tex         (~5 pp; includes Ethics/Env/Social subsection)
    ├── 06_conclusion.tex         (~2 pp; includes reflection paragraph)
    ├── A_risk_retrospective.tex  (≤1 page A4 — MANDATORY)
    ├── B_reproducibility.tex
    ├── C_code_listings.tex
    └── D_additional_overlays.tex
```

Approx page budget (main body, target ≤50 pp):
```
Intro 5 + Theory 12 + Apparatus 4 + Results 13 + Discussion 5
+ Conclusion 2 + References 2 + Appendices 5  ≈ 48 pp
```

## Writing schedule (5/25 → 6/1)

| Date | Sections | Pages added |
|---|---|---|
| 5/25 | skeleton + figure list | 0 (done) |
| 5/26 | Intro + Theory part 1 | ~9 |
| 5/27 | Theory part 2 + Apparatus | ~12 |
| 5/28 | Results: phantom + per-folio | ~7 |
| 5/29 | Results: GT + ablation + Discussion (with ethics) | ~11 |
| 5/30 | Conclusions + Technical Abstract + Risk appendix | ~5 |
| 5/31 | Language pass, refs, Appendix B/C/D, coversheet, PDF | polish only |
```
