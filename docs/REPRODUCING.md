# Reproducing the captioning results (Tables 1–2)

This page gives the exact protocol behind the paper's downstream captioning
numbers, the commands to rerun them on any GPU machine, and every known cause
of differences. The settings live in [`configs/eval_captioning.yaml`](../configs/eval_captioning.yaml).
They were read back from the logged configs of the runs that produced the
paper tables, not from memory.

Scope: Uniform, Random, Oracle and PEEK. CSTA and MaxInfo depend on
third-party code and weights and are not included.

## 1. Install

```bash
pip install -e ".[eval]"
```

You also need `java` on `PATH` (pycocoevalcap uses it for PTB tokenization
and METEOR). On aarch64, replace `decord` with `decord2`, which ships wheels
and provides the same `decord` module.

The paper runs used `transformers` 5.5.4 (ActivityNet Captions) and 5.6.2
(MSR-VTT), which give identical captions. The `eval` extra pins 5.5.4 and
`imageio-ffmpeg` 0.6.0 (ffmpeg 7.0.2, used to extract the paper's ANC frames).
See [§6](#6-known-causes-of-differences).

## 2. Data

**ActivityNet Captions (ANC).** The paper's "test" split is `val_1`
(17,505 segments in 4,885 videos with an available file, one reference each).
Build the manifest and extract 2 fps JPEG frames with the training scripts
(ffmpeg `-ss start -t duration -vf fps=2 -q:v 2`, per segment):

```bash
python scripts/prepare_manifest.py --annotations-root /path/to/anc/annotations \
    --videos-root /path/to/anc/videos --annotation-files val_1.json \
    --output-manifest data/manifests/anc_val1.jsonl
python scripts/extract_frames.py --manifest data/manifests/anc_val1.jsonl \
    --output-root data/anc_val1 --fps 2.0
```

**MSR-VTT.** The 2,990-clip test split with 20 references per clip. The paper
decoded the 3 fps, 298×224 copies in `MSRVTT_Videos.zip` from the Hugging Face
dataset [`friedrichor/MSR-VTT`](https://huggingface.co/datasets/friedrichor/MSR-VTT)
(revision `c1af215a96934854f42683c19c51391aaee6f962`). Use that copy: decoding
another encode of MSR-VTT gives different candidate frames.

```bash
python scripts/prepare_msrvtt_manifest.py --video-root data/msrvtt/videos \
    --output data/manifests/msrvtt_test.jsonl
```

## 3. Generate captions

One run per (dataset, VLM). `--vlm` is one of `smolvlm2`, `qwen25vl3b`,
`qwen25vl7b`, `qwen35_4b`. Runs are resumable, and `--shard i/n` splits the
items across GPUs or machines.

```bash
# ANC
python scripts/eval_captioning.py --dataset anc --vlm qwen25vl3b \
    --manifest data/manifests/anc_val1.jsonl --frames-root data/anc_val1/frames \
    --output runs/anc_qwen25vl3b.jsonl

# MSR-VTT, split over 4 GPUs
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/eval_captioning.py --dataset msrvtt --vlm qwen25vl3b \
      --manifest data/manifests/msrvtt_test.jsonl --video-root data/msrvtt/videos \
      --output runs/msrvtt_qwen25vl3b.shard$i.jsonl --shard $i/4 &
done; wait
```

Each output gets a `.meta.json` with the resolved settings and library
versions. Add `--limit 20` for a quick smoke test.

## 4. Score

```bash
python scripts/score_captions.py --dataset msrvtt \
    --manifest data/manifests/msrvtt_test.jsonl runs/msrvtt_qwen25vl3b.shard*.jsonl
```

This prints CIDEr, BLEU-4, METEOR and ROUGE-L (×100) per selector and k.
Failed or empty captions are left out, as in the paper, and the table shows
how many items were scored.

## 5. Protocol

| | ActivityNet Captions | MSR-VTT |
|---|---|---|
| Candidate frames | ffmpeg 2 fps JPEGs (`-q:v 2`) of the segment | decord, indices `round(arange(0, N, native_fps/2))` |
| Uniform | centre frame of each of k equal sub-segments | same |
| Random | one random frame per sub-segment, seed `sha256("0\|random_bucketed\|{video}\|{segment}")` | seed `sha256("0\|msrvtt_random_bucketed\|{video}\|{k}")` |
| Oracle | SigLIP2 cosine to the GT caption, **top-k** | mean SigLIP2 cosine over the 20 references, **stratified argmax** |
| PEEK | stratified argmax over **at most 32 frames** (linspace subsample) | stratified argmax over **all** frames |
| References | 1 per segment | 20 per clip |

Captioning, the same for both benchmarks: the k frames in temporal order in
one multi-image user message, then the prompt. bfloat16, SDPA attention,
greedy decoding (`do_sample=False`, `num_beams=1`). Everything else comes
from each checkpoint's `generation_config.json`, so Qwen2.5-VL keeps
`repetition_penalty=1.05`. `<think>` blocks and a leading `Assistant:` are
stripped from the output.

| VLM | Prompt | `max_new_tokens` | Image size |
|---|---|---|---|
| SmolVLM2-2.2B-Instruct | "Generate one short sentence (under 12 words) describing what is happening, like 'a man is playing guitar' or 'a woman is cooking'." | 50 | processor default |
| Qwen2.5-VL-3B-Instruct | "Describe this video in one sentence." | 50 | `qwen_vl_utils` default |
| Qwen2.5-VL-7B-Instruct | same as SmolVLM2 | 50 | `qwen_vl_utils` default |
| Qwen3.5-4B (`enable_thinking=False`) | "Write a single short caption (5–10 words) describing the main action in the video." | 20 | ANC: `max_pixels=401408`, 200704 on OOM. MSR-VTT: processor default |

The Qwen3.5 prompt has an en dash ("5–10") on ANC and an ASCII hyphen
("5-10") on MSR-VTT. The config keeps both.

## 6. Known causes of differences

These are what we found while reconstructing the runs. The first group is
built into the config. Change it only if you mean to depart from the paper.

**Built into the config (the paper ran this way):**
- **PEEK scored at most 32 frames on ANC.** The paper's ANC evaluation reused
  the training data pipeline, which subsamples segments longer than 32
  candidate frames to 32 with `np.linspace` (the training cap). This affects
  60% of ANC test segments. Uniform, Random and Oracle chose from every 2 fps
  frame, so on ANC PEEK chose from a coarser candidate set than the
  baselines. This contradicts the paper's statement that all methods use the
  same candidate frames. The selected frames are still 2 fps frames, and the
  cap matches how PEEK was trained (it never saw sequences longer than 32).
  MSR-VTT clips were scored uncapped (2 fps × ~15 s ≈ 30 frames, so rarely
  more than 32). `--peek-max-frames none` scores every ANC frame instead.
- **PEEK scoring precision.** On ANC the paper stored embeddings in float16
  and scored them under bfloat16 autocast, in padded batches of several
  segments. MSR-VTT used float32, one clip at a time. This code scores in
  float32 everywhere. On 60 ANC test segments, using the paper's own JPEG
  frames, float32 picks the same frames the paper captioned for 233/240 PEEK
  (segment, k) pairs. Emulating bf16 one segment at a time does worse,
  because the padded-batch rounding can't be recreated.
- **Oracle differs between benchmarks** (top-k on ANC, stratified argmax on
  MSR-VTT), and so do the Random seeds and the Qwen3.5 prompt punctuation and
  pixel cap.

**Environment (can move numbers slightly):**
- GPU and kernels. This is the main source of caption drift. ANC ran on
  NVIDIA A10G (torch 2.7.1), MSR-VTT on A100 (torch 2.11.0). bfloat16 greedy
  decoding is deterministic on a fixed machine, but not bit-identical across
  GPUs or torch builds: one near-tie in the logits changes a caption. On the
  same 60 ANC segments and the same frames, a rerun on another GPU (GB10)
  matched 67–79% of the paper's Qwen3.5 captions word for word, depending on
  the selector, and moved corpus CIDEr by up to ±4 per cell (60 segments
  with one reference each is a noisy sample). See [§8](#8-reproduction-check).
- `transformers` version. 5.5.4 and 5.6.2 gave identical captions (160/160)
  on the same GPU. Other releases can change chat templates or image
  processors. With templates that ignore `enable_thinking`, Qwen3.5 collapses
  to CIDEr ≈ 4. `eval_captioning.py` refuses to run Qwen3.5 when the
  template does not disable thinking.
- Video files and decoders. ANC frames depend on the ffmpeg build. With
  `imageio-ffmpeg` 0.8.x (ffmpeg 9) some segments get a different number of
  2 fps frames, which shifts every selector. They also depend on which ANC
  videos you can still download: a re-uploaded video gives different frames.
  MSR-VTT must be the 3 fps copy above, and the `decord` build may change
  PEEK and Oracle picks (see [§8](#8-reproduction-check)).

**Model checkpoints.** The released `momentslab/peek` weights are
bit-identical to the checkpoint used for the paper tables. VLM and SigLIP2
checkpoints are pulled from the Hub. Pin `revision=` if they get updated.

## 7. Notes on the published tables

We recomputed every Uniform, Random, Oracle and PEEK cell of Tables 1–2 from
the stored captions of the original runs. 115 of the 128 cells match to 0.01
CIDEr. The exceptions, and other details that matter when you compare your
numbers with the paper:

- **MSR-VTT, Qwen2.5-VL-7B (Oracle, Uniform, PEEK; 12 cells).** The captions
  of the run that produced these numbers are lost, so they cannot be
  rescored. A later rerun on the same frames came out about +1.6 CIDEr higher
  (the "generation drift" noted under Table 3), so expect a gap of that order
  for this row. The 7B Random row is from a surviving run and does match.
- **ANC, SmolVLM2, Random, k=1** is printed as 28.40. The stored captions
  give 28.38: the value was copied from a 3-decimal table.
- **SmolVLM2 returns empty captions** for about 40% of MSR-VTT clips at k=8,
  and for 114–131 ANC segments at k=8. On MSR-VTT the paper dropped them, so
  each selector at k=8 is scored on a different subset (n = 1,753 to 1,796).
  On ANC they were scored as empty strings, except for BLEU-4 in that row,
  which was computed without them (a difference of about 0.03).
  `score_empty_captions` in the config reproduces the CIDEr, METEOR and
  ROUGE-L behaviour.
- **ANC segment count.** The paper reports 17,505 test segments. Every run
  scored 17,504: `v_0dkIbKXXFzI / val1_seg_00002` (12.08–15.49 s, ending at
  the video's last timestamp) was never scored. At k=8, a few more segments
  (7, or up to 12 for Qwen2.5-VL-7B) have no caption and were skipped.

## 8. Reproduction check

We reran parts of the protocol with this code on a different GPU (NVIDIA
GB10, aarch64, torch 2.11) and compared with the paper's stored captions on
the same items. CIDEr ×100.

**MSR-VTT, Qwen2.5-VL-3B, first 600 test clips, Uniform and PEEK.**

| k | Uniform (paper) | Uniform (rerun) | PEEK (paper) | PEEK (rerun) |
|---|---|---|---|---|
| 1 | 31.02 | 30.41 | 29.45 | 29.59 |
| 2 | 33.37 | 34.64 | 35.24 | 34.68 |
| 4 | 41.61 | 41.05 | 41.42 | 41.57 |
| 8 | 42.80 | 43.33 | 44.52 | 43.44 |

Uniform picked the same frames in 100% of (clip, k) pairs and gave the same
caption in 55%. PEEK picked the same frames in 79%. Every cell is within
1.3 CIDEr of the paper, but on 600 clips the PEEK − Uniform gap moves by up
to 1.8 CIDEr, so compare gaps on the full test set. The PEEK frame
differences on MSR-VTT most likely come from decoding: this check used
`decord2` on aarch64, the paper `decord` on x86_64. We have not verified
this.

**ActivityNet Captions, Qwen3.5-4B, 60 segments, the paper's own JPEG
frames, all selectors.** Uniform, Random and Oracle picked the same frames in
240/240 (segment, k) pairs, PEEK in 233/240. Captions matched word for word
in 67% (Oracle) to 79% (Uniform) of pairs. Per-cell CIDEr moved by up to ±4,
which is expected on 60 segments with one reference each.
