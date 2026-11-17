#!/usr/bin/env bash
# Videos and filmstrips of the slides, from rollouts recorded on validation seeds:
#   MUJOCO_GL=egl uv run python scripts/evaluate_policy.py outputs/<run>/n300/seed<s>/selected.ckpt \
#       --object <object> --seeds validation --num-episodes 2 --video-episodes 2 \
#       --out-dir outputs/presentation/<clip>
# Clips: a_source, a_A = act_deterministic seed 2; e_A, e_E = act_augment seed 2;
# d_E = act_oracle_E seed 0. Run from the repository root: bash docs/presentation/make_videos.sh
set -euo pipefail
in=outputs/presentation
out=docs/presentation
mkdir -p $out/videos $out/figures

clip() { ls $in/$1/ep000_*.mp4; }

# Filmstrip: five evenly spaced frames of the wrist camera (right third of the tile).
strip() {
  local f n sel
  f=$(clip $1)
  n=$(ffprobe -v error -show_entries stream=nb_frames -of csv=p=0 "$f")
  sel=$(for i in 0 1 2 3 4; do printf "eq(n\\,%d)+" $((i * (n - 1) / 4)); done)
  ffmpeg -v error -y -i "$f" -vf "select='${sel%+}',crop=224:224:456:0,tile=5x1:padding=4:color=white" \
    -frames:v 1 $out/figures/strip_$1.png
}

# Video: clips stacked, each with its label; shorter clips hold their last frame (10 Hz).
video() {
  local name=$1 inputs=() filter="" i=0 n max=0
  shift
  while (($#)); do
    inputs+=(-i "$(clip $1)")
    n=$(ffprobe -v error -show_entries stream=nb_frames -of csv=p=0 "$(clip $1)")
    max=$((n > max ? n : max))
    filter+="[$i:v]tpad=stop_mode=clone:stop_duration=30,scale=1360:448:flags=lanczos,"
    filter+="drawtext=text='$2':x=12:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[v$i];"
    i=$((i + 1))
    shift 2
  done
  if ((i > 1)); then
    filter+="$(for j in $(seq 0 $((i - 1))); do printf "[v$j]"; done)vstack=inputs=$i:shortest=1[v]"
  else
    filter+="[v0]null[v]"
  fi
  ffmpeg -v error -y "${inputs[@]}" -filter_complex "$filter" -map "[v]" -frames:v $((max + 10)) \
    -c:v libx264 -pix_fmt yuv420p -crf 20 $out/videos/$name.mp4
}

for c in a_source a_A e_A e_E d_E; do strip $c; done
video task a_source "(a) Source only, on the source cube"
video appearance a_A "(a) Source only, on the blue cube A" e_A "(e) Augmentation, on the blue cube A"
video behaviour e_E "(e) Augmentation, on block E" d_E "(d) Oracle for E, on block E"

# The six objects at the start of an episode: wrist views of docs/scene.png in one row.
f=""
for i in 0 1 2 3 4 5; do f+="[0:v]crop=224:130:468:$((116 + 256 * i))[o$i];"; done
ffmpeg -v error -y -i docs/scene.png -filter_complex \
  "$f[o0][o1][o2][o3][o4][o5]xstack=inputs=6:layout=0_0|228_0|456_0|684_0|912_0|1140_0:fill=white" \
  -frames:v 1 $out/figures/objects.png
