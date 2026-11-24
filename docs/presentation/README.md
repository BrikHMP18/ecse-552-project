# Presentation (20%)

Group 9: **Wednesday December 2, 2026, 16:00-17:30** (lecture). 15 min + 5 min Q/A; every
member presents and is graded individually. Slides in PDF on myCourses by **November 29**;
own laptop.

Slides: [`main.tex`](main.tex) (Beamer), built with `make presentation` into
[`main.pdf`](main.pdf). Structure from [`../guidelines.md`](../guidelines.md): title,
outline, 20 content slides (target 15: trim before November 29), questions with links, 4
backup slides. Each slide has the section in the grey band and its main idea as the title.
Numbers come from `docs/experiments/`; parts that wait for results or for the team
(related work, generative-video examples) are red `TODO`s.

| Slides | Content (rubric) | Presenter (proposed, team to confirm) |
|---|---|---|
| 3-6 | Motivation: teleoperation cost, generative video, application, problem and R1-R6 (B1-B3) | Brik |
| 7-8 | Related work (TODO, team), data (B6) | Brik |
| 9-14 | Method built step by step: IDM, DANN, labeling and filter, ACT and the methods (B4, B5) | Zheng Ye, Myriam |
| 15-16 | Validation design; validation loss against success (metrics) | Myriam |
| 17-20 | Results: appearance against behaviour, R4 intervals, the videos of E, DANN (B7) | Gonzalo, Brik |
| 21-22 | Critique with fixes, conclusions (B8) | Brik |

Videos: slides 5, 17 and 19 show frames of a rollout and a "Video" button that opens
`videos/{task,appearance,behaviour,idm}.mp4` (the PDF cannot hold video: keep `videos/` next to
`main.pdf` on the laptop; if the viewer blocks the button, open the file). The rollouts are
illustrations on validation seeds, one per case; `make_videos.sh` rebuilds them. The ACT
figure (`figures/act.png`) and the teleoperation photo (`figures/aloha.png`) are from Zhao
et al. (2023), the DANN figure (`figures/dann_ganin.png`) from Ganin et al. (2016), cited on
their slides.

Must cover B1-B8: [course requirements](../course-requirements.md#presentation-20).
