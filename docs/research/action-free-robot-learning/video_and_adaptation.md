# Video learning and domain-adaptation evidence

## Which papers most directly precede our learning mechanism?

### Takeaway
BCO establishes inverse-model pseudo-labeling; Giving Robots a Hand and AMPLIFY are close robotic precedents with materially different data and policy-input contracts. These differences constrain the project's contribution and reproduction claims.

### Cited Findings
- **BCO:** Torabi, Warnell and Stone, IJCAI 2018. Sections 3–4 use agent-specific state transitions and action-labeled exploration to train an IDM, infer demonstration actions, and train BC. BCO(0) has no post-demonstration interaction; iterative variants add interaction. It is a mechanistic predecessor, not the exact fixed-source RGB/object-shift experiment. [Paper](https://arxiv.org/abs/1805.01954).
- **VPT:** Baker et al., NeurIPS 2022. Section 3 and Appendices D/E separate noncausal video labeling from causal policy execution. Minecraft actions and training scale differ from our robotics study. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c7008aff45b5d8f0973b23e1a22ada0-Abstract-Conference.html).
- **Giving Robots a Hand:** Kim, Wu and Finn, arXiv:2307.05959v1, 2023. Sections 4.1–4.3 use masked eye-in-hand video, an IDM trained on diverse action-labeled robot play, and BC on inferred human-video actions plus robot labels. Section 4.3 adds a binary grasp state estimated from the previous pseudoaction during human-video training. Appendix A.3 and object-generalization descriptions distinguish play coverage from narrow robot demonstrations. No conference venue or measured deployment-grasp source is asserted here. [Paper](https://arxiv.org/abs/2307.05959), [project](https://giving-robots-a-hand.github.io/).
- **AMPLIFY:** Collins, Cheng, Aneja, Wilcox, Joffe and Garg, arXiv:2506.14198v1, 2025. Section 2 specifies video observations/goals and robot images/proprioception/actions. Motion tokens come from point tracks; the forward model uses videos, and the IDM uses images, proprioception and motion. Section 3.2/Table 5 puts all LIBERO subsets in video training but only LIBERO-90 in action training. Appendix D.5 describes separate model-training stages; it is not a drop-in offline pseudo-label pipeline. [Paper](https://arxiv.org/abs/2506.14198), [official repository](https://github.com/pairlab/AMPLIFY).

### Inferences
- Require matched source-label budgets. If adding source play for motion coverage, give it to all appropriate comparisons and count its labels.
- Present object-geometry transfer with a strict image-only policy as the experiment's controlled setting, not as a first demonstration of robot learning from action-free videos.
- Source-only and unadapted-IDM comparisons are both essential: beating a weak pseudo-label baseline does not establish improvement over the policy that ignores B.

### Gaps
- No source-play sufficiency, target geometry or runtime feasibility has been demonstrated for this project. Reproduction of the cited systems is not implied by method-level comparisons.

## What supports adversarial alignment and what limits it?

### Takeaway
DANN is an implementable ingredient; successful target control remains an empirical hypothesis. Classification theory provides a caution about invariance, while action-supervised robotic adaptation must not be presented as satisfying our hidden-target-label protocol.

### Cited Findings
- **DANN:** Ganin et al., JMLR 17(59), 2016. Shared encoder, supervised source objective, domain discriminator and gradient reversal use unlabeled target observations. Continuous-action robotic IDM regression is our adaptation. [Paper](https://www.jmlr.org/papers/v17/15-239.html).
- **Zhao et al.:** Zhao, Tachet des Combes, Zhang and Gordon, ICML 2019. The counterexample and bounds show that feature invariance plus low source error need not ensure adaptation; conditional/label-distribution issues matter. This is classification theory, not a robotics guarantee. [Proceedings](https://proceedings.mlr.press/v97/zhao19a.html).
- **ADDA:** Tzeng, Hoffman, Saenko and Darrell, CVPR 2017. Separate source/target encoders and staged adaptation differ from DANN's shared-encoder gradient reversal. [Paper](https://arxiv.org/abs/1702.05464).
- **GDA:** Cheng, Ma, Chen, Mandlekar, Garrett and Xu, NeurIPS 2025. Sections 3–4 train with action-labeled target demonstrations and a proprioceptive policy. Joint observation/action OT is formulated; the implemented ground cost uses proprioceptive states in place of actions, and DTW sampling also uses proprioception. This is richer than appearance-only alignment, but not a zero-target-action baseline. [Project](https://ot-sim2real.github.io/), [paper](https://arxiv.org/abs/2509.18631).
- **Third-Person Imitation Learning:** its adversarial imitation procedure involves policy interaction/RL rather than the fixed offline BC protocol here. It is contextual literature, not an unchanged comparison. [Paper](https://arxiv.org/abs/1703.01703).

### Inferences
- Domain confusion is not a success metric. Measure execution and preserve motion/gripper-relevant information; interpret harmful alignment as negative transfer.
- A source-based hyperparameter rule cannot establish target-optimal alignment. Predeclare a small sensitivity grid and report target evaluation after selection; disclose any target-development success used for feasibility decisions.
- A supervised B reference is an empirical additional-label comparison, not a guaranteed upper bound.

### Gaps
- No published result inspected here establishes which alignment location wins for the proposed OCBench geometry change.

## Which recent directions deserve limited supporting coverage?

### Takeaway
Structured visual-motion decoding is a useful possible extension. Large latent/video architectures should inform terminology and assumptions without becoming mandatory implementations.

### Cited Findings
- **VERA:** Li, Kim, Bai, Zhao, Pang, Simchowitz and Sitzmann, arXiv:2605.27817v1, May 2026. Sections 3.2–3.3 learn an image-conditioned Jacobian field using visual motion and action-labeled robot experience, then invert it with regularization to follow predicted video motion. Its large video planner differs from target-video pseudo-labeling. [Paper](https://arxiv.org/abs/2605.27817), [project](https://vera.csail.mit.edu/).
- LAPA, MotoVLA, UniVLA and DreamZero have source-grounded supervision/cost notes in the companion benchmark-baseline review. [LAPA](https://arxiv.org/html/2410.11758v2), [MotoVLA](https://arxiv.org/html/2509.19958v1), [UniVLA](https://arxiv.org/html/2505.06111v1), [DreamZero](https://arxiv.org/html/2602.15922v1).

### Inferences
- Read VERA if considering an interpretable motion-to-action extension after the main IDM experiment. Do not interpret its architecture as proof of target-object generalization here.
- Use LAPA to distinguish action-free pretraining from downstream labeled adaptation, and MotoVLA to distinguish video-seen/action-unseen from completely unseen tasks.

### Gaps
- Video Generators are Robot Policies was inspected during screening, but a sufficiently detailed supervision audit was not completed for inclusion as a recommended baseline. CORE (2606.29517) and ZimaBlue (2609.00188) were abstract-screened only. No exhaustive literature coverage or current-best-method claim is made.
- Evidence assembled 7 October 2026 from primary papers and code. No experiments or executable model checks were performed.
