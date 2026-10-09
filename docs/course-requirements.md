# ECSE 552 project requirements

Summarized from the instructor's project instructions (recorded Sep 1, 2026).
Check MyCourses for the current version before each submission.

Project = **50%** of the course grade. Team of exactly four.
Late work loses **10% per started 10-hour interval**.

| Deliverable | Weight | Due | Folder |
|---|---:|---|---|
| Proposal | 5% | Oct 11, 2026 (extended from Oct 7) | [`proposal/`](proposal) |
| Presentation | 20% | Last 1–2 weeks of class (slot TBD) | [`presentation/`](presentation) |
| Final report | 20% | Dec 6, 2026, 23:59 | [`report/`](report) |
| Two peer reviews | 5% | Dec 13, 2026, 23:59 | — |

Rules that apply to every deliverable:

- Baselines and alternatives (including SOTA) **must be run on our own data**;
  numbers copied from papers do not count.
- Proposal and report require citations.
- Any generative-AI use must follow McGill's research guidance and be
  **explicitly disclosed and justified** — log it in [`ai-use-log.md`](ai-use-log.md).

## Proposal (5%)

About **two pages, under 1,000 words**, plus one extra page for references.
Use these six headings, with visible subheadings (e.g. "Constraints") so graders
can find each A-element.

| ID | Assessed element | Heading |
|---|---|---|
| A1 | Define the application | Background |
| A2 | Problem and why a solution is needed | Background |
| A3 | Application-specific design constraints | Background; Aims/Goals |
| A4 | Project scope | Aims/Goals |
| A5 | Proposed solution | Aims/Goals; Methodology |
| A6 | Implementation plan | Methodology |
| A7 | Validation, testing, evaluation | Methodology |
| A8 | Alternatives and why ours could be superior | Methodology |
| A9 | Feasibility, including the dataset | Feasibility and resources |
| A10 | Risks, pitfalls, mitigations | Risk analysis |
| — | Explicit division of work | Role of group members |

Constraints are requirements the design must satisfy (accuracy, latency, input
symmetries…), not team time or equipment access.

## Presentation (20%)

Fewer than ~1 slide per minute, slide numbers, figures over text.

| ID | Content | Section |
|---|---|---|
| B1 | Define the application | Introduction/Motivation |
| B2 | Problem and why a solution was needed | Problem Definition |
| B3 | Important design constraints | Problem Definition |
| B4 | Designed solution | Methodology |
| B5 | Implementation factors | Methodology |
| B6 | Datasets and resources | Methodology |
| B7 | Results vs. alternatives | Results/Evaluations |
| B8 | How B3 constraints were met; what went wrong | Conclusions |

Rubric (100): slides 5 · flow 5 · intro + related work 10 · problem + constraints 10 ·
datasets 5 · solution/architecture/implementation 20 · rigorous metrics 10 ·
comparison with baselines 15 · conclusions/limitations 10 · delivery 10.

## Final report (20%)

NeurIPS or Oxford *Bioinformatics* style; **~5,000–8,000 words** excluding
references. The main report must be self-contained.

Structure: Abstract · Introduction (C1, C2 high-level, C3, C4) · Methods (C2
detailed, C5–C8) · Results (C9, C10) · Discussion and Conclusions (C4 met?, C11) ·
References.

| ID | Required element |
|---|---|
| C1 | Define the application |
| C2 | Formulate the problem |
| C3 | Why a solution was needed |
| C4 | Constraints important to the application |
| C5 | Resources and datasets |
| C6 | Designed solution and implementation, in detail |
| C7 | Evaluation approach, in detail |
| C8 | Alternative solutions |
| C9 | Quantitative results and how the design meets C4 |
| C10 | Quantitative comparison with alternatives on C4 |
| C11 | Limitations and advantages |

Rubric (100): abstract 5 · intro/motivation 10 · related work 5 · constraints +
formulation 10 · detailed formulation 5 · datasets 5 · solution 10 · splits/metrics 10 ·
baselines 5 · results vs. constraints and alternatives 15 · limitations/advantages 10 ·
writing 5 · figures/tables 5.
