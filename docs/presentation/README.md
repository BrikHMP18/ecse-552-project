# Presentation (20%)

Group 9: **Wednesday December 2, 2026, 16:00-17:30** (lecture). 15 min + 5 min Q/A; every
member presents and is graded individually. Slides in PDF on myCourses by **November 29**;
own laptop.

Slides: [`main.tex`](main.tex) (Beamer), built with `make presentation` into
[`main.pdf`](main.pdf). Structure from [`../guidelines.md`](../guidelines.md): title,
outline, 15 content slides, questions with links, 5 backup slides. Numbers come from
`docs/experiments/` and the report; parts that wait for results are red `TODO`s.

| Slides | Content (rubric) | Presenter (proposed, team to confirm) |
|---|---|---|
| 3-6 | Motivation and application, problem and constraints, existing approaches, data (B1-B3, B6) | Brik |
| 7-9 | ACT and what each method trains, solution, IDM (B4, B5) | Zheng Ye |
| 10-11 | DANN, validation design (B5, metrics) | Myriam |
| 12-13 | Results 1-2: appearance against behaviour, comparison with R4/R5 intervals (B7) | Gonzalo |
| 14 | Result 3: the videos of E | Brik |
| 15 | Result 4: DANN makes the labels worse | Myriam |
| 16-17 | Constraints met, what went wrong, conclusions and limitations (B8) | Brik |

Videos: slides 3, 12 and 14 show frames of a rollout and a "Video" button that opens
`videos/{task,appearance,behaviour,idm}.mp4` (the PDF cannot hold video: keep `videos/` next to
`main.pdf` on the laptop; if the viewer blocks the button, open the file). The rollouts are
illustrations on validation seeds, one per case; `make_videos.sh` rebuilds them. The ACT
figure (`figures/act.png`) is from Zhao et al. (2023), cited on the slide.

Must cover B1-B8: [course requirements](../course-requirements.md#presentation-20).
