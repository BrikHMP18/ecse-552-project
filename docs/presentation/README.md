# Presentation (20%)

Group 9: **Wednesday December 2, 2026, 16:00-17:30** (lecture). 15 min + 5 min Q/A; every
member presents and is graded individually. Slides in PDF on myCourses by **November 29**;
own laptop.

Slides: [`main.tex`](main.tex) (Beamer), built with `make presentation` into
[`main.pdf`](main.pdf). 15 slides and 3 backup slides; numbers come from the report. Parts
that wait for the IDM and DANN results are red `TODO`s.

| Slides | Content (rubric) | Presenter (proposed) |
|---|---|---|
| 2-6 | Motivation, application, problem, constraints, data | Brik |
| 7-8 | Pipeline, IDM | Zheng Ye |
| 9 | DANN | Myriam |
| 10-11 | Policy, alternatives, evaluation, augmentation result | Gonzalo |
| 12-15 | Target E, comparison, constraints, conclusions | Brik |

Videos: slides 3, 11 and 12 show frames of a rollout and a "Video" button that opens
`videos/{task,appearance,behaviour}.mp4` (the PDF cannot hold video: keep `videos/` next to
`main.pdf` on the laptop; if the viewer blocks the button, open the file). The rollouts are
illustrations on validation seeds, one per case; `make_videos.sh` rebuilds them.

Must cover B1-B8: [course requirements](../course-requirements.md#presentation-20).
