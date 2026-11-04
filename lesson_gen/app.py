"""Teacher-facing UI.   python app.py   (set LESSON_ADAPTER=adapters/lesson-lora to use the fine-tuned model)"""
import os, tempfile
import gradio as gr
from pipeline import run, TemplateBackend, HFBackend, HFAPIBackend, to_markdown, to_docx

_cache = {}
BASE = os.environ.get("LESSON_BASE", "Qwen/Qwen2.5-3B-Instruct")
API_BASE = os.environ.get("LESSON_API_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_ADAPTER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lesson-lora-new"))
ADAPTER = os.environ.get("LESSON_ADAPTER") or (DEFAULT_ADAPTER if os.path.isdir(DEFAULT_ADAPTER) else None)
DEMO = "Demo (offline templates - no AI model)"
FT = "Fine-tuned model" if ADAPTER else f"Small hosted model ({API_BASE})"


def backend(mode):
    if mode not in _cache:
        if mode == DEMO:
            _cache[mode] = TemplateBackend()
        else:
            try:
                if ADAPTER:
                    _cache[mode] = HFBackend(BASE, ADAPTER)
                else:
                    token = os.environ.get("HF_TOKEN")
                    if not token:
                        raise RuntimeError("HF_TOKEN is required for the hosted model")
                    _cache[mode] = HFAPIBackend(API_BASE, token)
            except Exception as e:      # missing torch/transformers/GPU/adapter -> clear message, not a crash
                raise gr.Error(f"Could not load the model ({type(e).__name__}: {e}). "
                               "Check the Space dependencies/model access, or choose Demo mode.")
    return _cache[mode]


def generate(grade, subject, topic, keywords, mode, retries):
    if not topic.strip() or not keywords.strip():
        raise gr.Error("Please enter a topic and at least one keyword.")
    import time
    t0 = time.time()
    out = run(backend(mode), int(grade), subject, topic.strip(), keywords, max_retries=int(retries))
    print(f"[generate] mode={mode} took {time.time() - t0:.0f}s, attempts={len(out['trace'])}, "
          f"story_words={len(out['story'].split())}, raw_chars={len(out['raw'])}")
    if not out["story"] and out["raw"].strip():     # model broke the STORY:/LESSON PLAN: format -> still show its text
        out["story"] = "(The model did not follow the expected format, so its raw output is shown here.)\n\n" + out["raw"]
    elif not out["raw"].strip():
        raise gr.Error("The model returned empty text. Check the adapter matches the base model, or use Demo mode.")
    md = to_markdown(out["plan"], out["story"], out["lesson"])
    d = tempfile.mkdtemp()
    path = os.path.join(d, "lesson.docx")
    files = to_docx(out["plan"], out["story"], out["lesson"], path)
    if not files:
        path = os.path.join(d, "lesson.md"); open(path, "w").write(md)
    status = ("Passed all checks" if out["passed"] else "Needs teacher review: " + "; ".join(out["issues"]))
    if out["dropped"]:
        status += f"\nSafety guard removed unsuitable keyword(s): {', '.join(out['dropped'])}"
    report = f"{status}\n\nMetrics: {out['metrics']}\n\nAgent trace:\n" + "\n".join(
        f"  attempt {t['attempt']}: {len(t['issues'])} issue(s) {t['issues']}" for t in out["trace"])
    return out["story"], out["lesson"], report, path


with gr.Blocks(title="AI Teacher Story & Lesson-Plan Assistant") as demo:
    gr.Markdown("# AI Teacher Story & Lesson-Plan Assistant\nGenerate age-appropriate stories and structured lesson plans using a fine-tuned language model. "
                "The planner, fine-tuned generator and critic agents draft a story and a lesson plan. Always review before class.")
    with gr.Row():
        grade = gr.Dropdown([1, 2, 3, 4, 5, 6, 7, 8], value=3, label="Grade")
        subject = gr.Dropdown(["Science", "Math", "EVS", "Social Science", "Values", "English"], value="Science", label="Subject")
    topic = gr.Textbox(label="Topic", placeholder="Water cycle")
    keywords = gr.Textbox(label="Keywords (comma separated)", placeholder="rain, river, clouds")
    with gr.Row():
        mode = gr.Radio([DEMO, FT], value=FT if ADAPTER else DEMO, label="Generator")
        retries = gr.Slider(0, 3, value=2, step=1, label="Max critic retries")
    btn = gr.Button("Generate", variant="primary")
    story = gr.Textbox(label="Story", lines=10)
    lesson = gr.Textbox(label="Lesson plan", lines=16)
    report = gr.Textbox(label="Critic report", lines=5)
    file = gr.File(label="Download")
    btn.click(generate, [grade, subject, topic, keywords, mode, retries], [story, lesson, report, file])

if __name__ == "__main__":
    demo.launch(
        share=os.environ.get("GRADIO_SHARE", "false" if os.environ.get("SPACE_ID") else "true").lower() == "true",
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        footer_links=[],
    )
