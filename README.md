# ECSE 552 Project — Learning to Manipulate a New Object from Action-Free Videos

McGill ECSE 552 (Deep Learning), Fall 2026 · Team-review draft

Use [OCBench](https://github.com/seohongpark/ocbench) directly for simulation and
demonstration generation, retaining its UR5e arm and Robotiq gripper. The proposed
experiment starts from its visual single-block `lite` task and adds a short
elongated-block variant. Learn from labeled cube demonstrations and **image-only** block
videos. Test whether adversarial adaptation of an inverse dynamics model improves
pseudoactions and closed-loop policy success. The learned policy also uses only
images; shared low-level control may use robot state.

## Simulation and data plan

- Start from `visual-block-cpu-lite-single-task1-v0` (CPU MuJoCo).
  MJWarp acceleration is optional after the CPU pilot works.
- Retain the existing seven normalized actions: six joint increments and one
  gripper-opening increment. Use front, side, and wrist RGB observations;
  exclude privileged `info` fields from all learned-model inputs.
- Generate a small controlled dataset with `observation_interval=1` so each
  control step has an image. The published visual collection commands use sparse
  images; those files are not a drop-in dataset for our transition-based IDM.
- Extend cube geometry and expert grasp logic for the target block. Validate
  successful demonstrations before training; this extension is not implemented.
- Extend positional success to require a lift, release, and one second of stable
  placement. Disable the stock early-success termination so this can be measured.
- Adding Panda to OCBench is a possible later extension, not a project dependency.

Inspection baseline: upstream commit `e2cd2f72110b66bd65afab1b855d81ebc73aeacc`.
OCBench has been reviewed at source level; no runtime pilot has been run here.

The scientific protocol remains a proposal for team review and validation.
The current draft is `proposal/main.tex` / `proposal/main.pdf`. The previous
ALOHA/ACT outline by Gonzalo is preserved in `proposal/archive/`; the final report
remains a scaffold with no experimental results.

## Decisions for team review

- Accept or revise the single-arm object-change task and image-only input contract.
- Confirm GPU access and use the pilot to budget training and evaluation.
- Assign simulation/data, policy/baselines, IDM/adaptation, and evaluation/integration.
- Confirm the roster: the existing cover lists Myriam Dardour, Brik Meza Pinedo,
  Peizhe Tian, and Zheng Ye Zhang; the earlier responsibility table listed Gonzalo
  instead of Peizhe. The cover names are preserved pending team confirmation.
- Verify the scientific claims, bibliography, and AI disclosure before submission.

## Deliverables

Literature selection and experimental recommendations: [Action-free robot learning review](research_notes/action-free-robot-learning/review.md) (7 October 2026; primary-source review, no experiments).
Baseline selection (top 5 comparisons, GDA verdict, novelty check): [baseline-selection.md](research_notes/action-free-robot-learning/baseline-selection.md) (8 October 2026; verified against primary sources, no experiments).

| Deliverable | Weight | Due | Source | PDF |
|---|---:|---|---|---|
| Proposal | 5% | Oct 11 (extended from Oct 7) | [`proposal/main.tex`](proposal/main.tex) | [`proposal/main.pdf`](proposal/main.pdf) |
| Presentation | 20% | Last 1–2 weeks of class | [`presentation/`](presentation) | — |
| Final report | 20% | Dec 6, 23:59 | [`report/main.tex`](report/main.tex) | [`report/main.pdf`](report/main.pdf) |
| Peer reviews | 5% | Dec 13, 23:59 | — | — |

Requirements and rubrics for each: [`docs/course-requirements.md`](docs/course-requirements.md).

## Layout

```
proposal/        proposal (≈2 pages, <1,000 words + 1 page of references)
report/          final report, NeurIPS 2026 style (5,000–8,000 words)
presentation/    slides
docs/            course requirements, generative-AI use log
references.bib   shared bibliography for all documents
```

Code (simulation, IDM, policy, evaluation) will be added as the project starts.

## Writing format

Proposal and report use the official **NeurIPS 2026** LaTeX style in `preprint`
mode, with visible authors. The unmodified `neurips_2026.sty` files come from the
[official author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip),
linked by the [2026 call for papers](https://neurips.cc/Conferences/2026/CallForPapers).
Do not override its margins, fonts, title layout, or heading spacing. The only change is removing the "Preprint." notice at the
foot of page 1, since these are course documents.

The course determines the document structure: the proposal retains its six required
sections and fewer than 1,000 words; the report retains Abstract, Introduction,
Methods, Results, Discussion and Conclusions, and References. This is a course
project using the conference's typography, not a NeurIPS submission. The proposal's
A1–A10 mapping is kept in LaTeX comments so its prose reads naturally.

## Building

```bash
make            # proposal + report, PDF and Word
make proposal   # one document (PDF)
make docx       # Word copies only (proposal/main.docx, report/main.docx)
make clean      # remove LaTeX build files
```

`main.docx` is a one-way export for sharing on Google Drive (requires
[pandoc](https://pandoc.org)). LaTeX stays the source: copy any edits made in
Drive back into `main.tex`. The Word copy approximates the PDF without its exact
layout: Times New Roman 10 pt, justified paragraphs, numbered
sections, and numbered citations that link to the reference list (IEEE style from
`tools/ieee.csl`, CC BY-SA 3.0, from the
[CSL styles repository](https://github.com/citation-style-language/styles)). Text is
black except the team's notes (`[TEAM: ...]`, `[TO WRITE: ...]`), which stay red
as in the PDF. The template `tools/reference.docx` is
built by `tools/make_reference_docx.py`; `tools/docx_fix_bookmarks.py` keeps only
the bookmarks citations link to, with visible names, so links survive Google Docs
import without a marker icon on every heading.

Commit the rebuilt PDF together with any `.tex` change.

## Generative AI

The course requires disclosing and justifying any generative-AI use. Log every
material use in [`docs/ai-use-log.md`](docs/ai-use-log.md).
