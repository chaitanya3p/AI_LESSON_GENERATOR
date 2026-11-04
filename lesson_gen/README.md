---
title: AI Story & Lesson-Plan Generator
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
models:
  - Qwen/Qwen2.5-3B-Instruct
---

# AI Story & Lesson-Plan Generator for Teachers
Fine-tuned LLM (QLoRA on Qwen2.5-3B-Instruct) + agentic Planner -> Generator -> Critic -> Formatter pipeline.

**Abstract.** This project fine-tunes a generative language model (Qwen2.5-3B with QLoRA) to draft age-appropriate
stories and lesson plans for teachers from a grade, subject, topic and keywords. Curriculum-aligned seed topics
for grades 1-8 (32 topics, 6 subjects) are expanded, de-duplicated, quality-gated and split by topic into training
and unseen-topic test sets. The model is compared against its base version, with and without a critic loop that
checks structure, keyword coverage, readability and child-safety, and is evaluated with automatic metrics,
a safety/bias probe set and teacher ratings. The outcome is a Gradio lesson-planning assistant with Word export.

**Keywords:** story generation, lesson plan, teachers, fine-tuned language model, QLoRA, agentic AI.

## Files
| File | Purpose |
|---|---|
| `curriculum.py` | 32 curriculum seed topics (grades 1-8): objectives, facts, activities, questions |
| `build_dataset.py` | seeds -> 3 story structures x lesson variants -> de-duplicate -> critic quality gate -> topic-held-out split -> `data/data_report.json` |
| `data/` | `train`, `val`, `test` (seen topics), `test_unseen` (6 topics never trained on), `data_report.json` |
| `train_qlora.py` | QLoRA fine-tuning; saves adapter and `results/train_log.json` (loss curves) |
| `pipeline.py` | Planner (with unsafe-keyword input guard), backends, Critic, retry loop, Markdown/DOCX export |
| `evaluate.py` | structure, keywords, readability, safety, pass rate, ROUGE-L, distinct-2, per-grade pass rate -> `results/*.json` |
| `safety_probes.py` | unsafe-keyword, out-of-curriculum and gender-pronoun probes |
| `human_eval.py` | blind teacher rating sheet + summary with inter-rater agreement |
| `compare_results.py` | merges all runs into one report table |
| `run_experiments.sh` | one command to run everything on a GPU |
| `test_pipeline.py` | offline unit tests (`python test_pipeline.py`) |
| `leak_check.py` | contamination check: the adapter must not reproduce curriculum sentences of the held-out topics |
| `make_report.py` | builds `results/RESULTS_SECTION.md` (tables + factual observations) from `results/*.json` |
| `colab_run_all.ipynb` | one notebook that unpacks, checks, evaluates and reports (run top to bottom on a T4) |
| `make_small_adapter.ps1` | Windows script: turns the downloaded `lesson-lora.zip` into a small `lesson-lora-new.zip` |

## Run
```bash
pip install -r requirements.txt
python build_dataset.py          # regenerates data/ and data_report.json
python test_pipeline.py
python app.py                    # uses adapters/lesson-lora automatically if present, else demo mode
```

## Permanent Render deployment

The free Render configuration uses the same Gradio UI and sends generation to the
small `Qwen/Qwen2.5-0.5B-Instruct` model through the Hugging Face Inference API,
avoiding local model-memory limits. Create a Render Web Service from this folder
using `render.yaml` and add a Hugging Face read token as the `HF_TOKEN` secret.
The existing local QLoRA adapter remains available by setting `LESSON_ADAPTER`
and `LESSON_BASE` together; a LoRA adapter must use its matching base model.

The project’s fine-tuning, dataset construction, critic, safety guard, and
evaluation remain in the repository. The Render free tier may sleep when idle,
but its public URL is persistent.

## Fine-tune and evaluate (Colab/Kaggle T4)
```bash
pip install -r requirements-train.txt
bash run_experiments.sh          # builds data, trains, runs 3 experiments x 2 test sets, probes, human sheet
python compare_results.py > results/comparison.md
```
Experiments: (1) base model, (2) fine-tuned, (3) fine-tuned + agent loop, each on `test` and `test_unseen`.
Then give `results/human_ratings_blank.csv` to 3-5 teachers and run `python human_eval.py summarize results/human_ratings.csv`.

**Important:** the dataset was rebuilt (32 topics, 1,920 examples). An adapter trained on the older 16-topic data
must be retrained with `train_qlora.py` before experiments 2 and 3.

## Limitations (state these in the report)
- Seed topics are hand-written, NCERT-style learning outcomes, not copied syllabus text; verify them against the current syllabus.
- Training targets are template-generated (3 story structures), so the model learns the format and style, not new pedagogy.
  The offline demo backend re-uses the generator, so its scores are not meaningful evidence of model quality.
- Readability uses Flesch-Kincaid with a grade + 4 limit because subject words inflate the score.
- Output always needs teacher review.
