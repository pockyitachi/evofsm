# paper_draft/ — the EvoFSM-RL paper (working draft)

LaTeX source for the EvoFSM-RL paper, one file per section. There is **no
`main.tex`** — assemble it per target venue.

| File | Section |
|---|---|
| `abstract.tex` | Abstract |
| `intro.tex` | Introduction (motivation anchored on arXiv 2603.07432) |
| `related.tex` | Related work |
| `method.tex` | Method (two-layer FSM + joint loop + GRPO, benchmark-agnostic) |
| `experiment.tex` | Experiments — one chapter, two levels (within-benchmark AW+, cross-benchmark AW+→MobileWorld) |
| `limitation.tex` | Limitations |
| `conclusion.tex` | Conclusion |
| `appendix.tex` | Appendix |
| `reference.bib` | Bibliography (21 entries) |

## Usage
Add your own `main.tex` with the preamble, `\input{}` of these sections, and
`\bibliography{reference}`. Cross-references and citations are self-consistent
(every `\ref` has a `\label`, every `\cite` is in the bib).

See `CLAUDE.md` here for the paper's framing and the data-number discipline
(provisional placeholders, locked lineages).
