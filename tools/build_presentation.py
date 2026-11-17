from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
OUTPUT = ROOT / "Coal_Detection_Project_Presentation.pptx"

INK = RGBColor(29, 36, 34)
MUTED = RGBColor(100, 112, 106)
PAPER = RGBColor(243, 241, 234)
WHITE = RGBColor(251, 250, 246)
FOREST = RGBColor(47, 77, 67)
MINT = RGBColor(183, 212, 190)
AMBER = RGBColor(230, 161, 74)
CORAL = RGBColor(215, 107, 82)
LINE = RGBColor(215, 216, 206)

FONT = "Aptos"
MONO = "Aptos Mono"


def set_background(slide, color=PAPER):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_box(slide, x, y, w, h, fill=None, line=None, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill or PAPER
    shape.line.color.rgb = line or fill or PAPER
    if radius:
        shape.adjustments[0] = 0.05
    return shape


def add_text(slide, text, x, y, w, h, size=18, color=INK, bold=False,
             font=FONT, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
             margin=0, italic=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.text_frame.clear()
    box.text_frame.word_wrap = True
    box.text_frame.margin_left = Inches(margin)
    box.text_frame.margin_right = Inches(margin)
    box.text_frame.margin_top = Inches(margin)
    box.text_frame.margin_bottom = Inches(margin)
    box.text_frame.vertical_anchor = valign
    paragraph = box.text_frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return box


def add_rich_text(slide, runs, x, y, w, h, size=18, color=INK, font=FONT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.text_frame.clear()
    box.text_frame.word_wrap = True
    paragraph = box.text_frame.paragraphs[0]
    for text, bold, run_color in runs:
        run = paragraph.add_run()
        run.text = text
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = run_color or color
    return box


def add_footer(slide, number, dark=False):
    color = RGBColor(210, 218, 210) if dark else MUTED
    add_text(slide, "COAL DETECTION / NIR STUDY", 0.55, 7.12, 4.5, 0.18,
             size=8, color=color, font=MONO)
    add_text(slide, f"{number:02d} / 19", 12.1, 7.12, 0.75, 0.18,
             size=8, color=color, font=MONO, align=PP_ALIGN.RIGHT)


def add_eyebrow(slide, text, x=0.65, y=0.55, dark=False):
    add_text(slide, text.upper(), x, y, 7.0, 0.22, size=9,
             color=AMBER if dark else CORAL, font=MONO)


def add_heading(slide, title, x=0.65, y=0.9, w=8.8, size=31, color=INK):
    add_text(slide, title, x, y, w, 0.9, size=size, color=color, bold=True)


def add_notes(slide, notes):
    slide.notes_slide.notes_text_frame.text = notes


def add_image(slide, path, x, y, w, h):
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def add_pill(slide, text, x, y, w, fill=MINT, color=FOREST):
    add_box(slide, x, y, w, 0.3, fill=fill, line=fill, radius=True)
    add_text(slide, text.upper(), x, y + 0.03, w, 0.18, size=8, color=color,
             font=MONO, align=PP_ALIGN.CENTER)


def add_bullet_list(slide, items, x, y, w, line_height=0.52, size=16, color=INK):
    for index, item in enumerate(items):
        yy = y + index * line_height
        add_text(slide, "•", x, yy, 0.22, 0.25, size=size, color=CORAL, bold=True)
        add_text(slide, item, x + 0.3, yy, w - 0.3, 0.38, size=size, color=color)


def add_metric_card(slide, value, label, x, y, w=2.7, fill=WHITE, value_color=FOREST):
    add_box(slide, x, y, w, 1.15, fill=fill, line=LINE)
    add_text(slide, value, x + 0.18, y + 0.18, w - 0.36, 0.45, size=27,
             color=value_color, bold=True)
    add_text(slide, label.upper(), x + 0.18, y + 0.76, w - 0.36, 0.2,
             size=8, color=MUTED, font=MONO)


def add_pipeline_step(slide, number, title, subtitle, x, y, w, fill=WHITE, text_color=INK):
    add_box(slide, x, y, w, 1.38, fill=fill, line=LINE)
    add_text(slide, number, x + 0.18, y + 0.15, w - 0.36, 0.2, size=9,
             color=CORAL if fill != CORAL else WHITE, font=MONO)
    add_text(slide, title, x + 0.18, y + 0.62, w - 0.36, 0.3, size=16,
             color=text_color, bold=True)
    add_text(slide, subtitle, x + 0.18, y + 1.02, w - 0.36, 0.2, size=8,
             color=MUTED if fill != CORAL else WHITE, font=MONO)


def add_bar(slide, label, value, x, y, w=5.0, color=FOREST):
    add_text(slide, label, x, y, 1.15, 0.2, size=9, color=MUTED, font=MONO)
    add_box(slide, x + 1.25, y + 0.05, w, 0.12, fill=LINE, line=LINE)
    add_box(slide, x + 1.25, y + 0.05, w * value / 100, 0.12, fill=color, line=color)
    add_text(slide, f"{value:.2f}%", x + 1.25 + w + 0.15, y - 0.02, 0.75, 0.2,
             size=9, color=INK, font=MONO, align=PP_ALIGN.RIGHT)


def make_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Title
    slide = prs.slides.add_slide(blank)
    set_background(slide, PAPER)
    add_eyebrow(slide, "NIR SPECTROSCOPY / MODEL STUDY")
    add_text(slide, "Coal Detection\nfrom NIR Spectroscopy", 0.65, 1.25, 6.5, 1.65,
             size=40, bold=True)
    add_text(slide, "A material-aware classification study for coal and coal-measure rocks.",
             0.68, 3.15, 4.8, 0.6, size=17, color=MUTED)
    add_pill(slide, "Selected stable pipeline", 0.68, 4.08, 2.15)
    add_text(slide, "StandardScaler  ->  PCA (2)  ->  RBF SVM", 0.68, 4.58, 5.7, 0.35,
             size=17, color=FOREST, bold=True, font=MONO)
    add_box(slide, 8.0, 1.05, 4.55, 4.8, fill=INK, line=INK)
    add_text(slide, "RAW NIR SPECTRUM", 8.45, 1.5, 3.5, 0.25, size=9, color=AMBER, font=MONO)
    add_pipeline_step(slide, "01", "StandardScaler", "training statistics", 8.45, 2.0, 3.65, fill=WHITE)
    add_text(slide, "↓", 10.1, 3.4, 0.4, 0.3, size=20, color=AMBER, align=PP_ALIGN.CENTER)
    add_pipeline_step(slide, "02", "PCA / 2 components", "99.472% variance", 8.45, 3.8, 3.65, fill=WHITE)
    add_text(slide, "→  COAL / NON-COAL", 8.45, 5.45, 3.65, 0.25, size=10, color=MINT, font=MONO)
    add_footer(slide, 1)
    add_notes(slide, "Introduce the project as an NIR spectroscopy classifier. Emphasize that the selected pipeline is the most stable according to repeated material-grouped validation, not an absolute claim of universal superiority.")

    # 2. Objective
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_eyebrow(slide, "01 / THE OBJECTIVE")
    add_heading(slide, "Can a spectrum carry enough evidence to identify coal?")
    add_text(slide, "The system reads one measured NIR spectrum and assigns one of two labels.", 0.68, 2.0, 6.7, 0.45, size=20, color=MUTED)
    add_box(slide, 0.68, 3.0, 5.55, 1.65, fill=FOREST, line=FOREST)
    add_text(slide, "01", 1.0, 3.28, 0.6, 0.25, size=10, color=AMBER, font=MONO)
    add_text(slide, "COAL", 1.0, 3.72, 2.7, 0.45, size=30, color=WHITE, bold=True)
    add_text(slide, "label 1", 4.7, 3.83, 0.95, 0.2, size=10, color=MINT, font=MONO)
    add_box(slide, 6.72, 3.0, 5.55, 1.65, fill=WHITE, line=LINE)
    add_text(slide, "00", 7.04, 3.28, 0.6, 0.25, size=10, color=CORAL, font=MONO)
    add_text(slide, "NON-COAL", 7.04, 3.72, 3.3, 0.45, size=30, color=INK, bold=True)
    add_text(slide, "label 0", 11.0, 3.83, 0.95, 0.2, size=10, color=MUTED, font=MONO)
    add_text(slide, "The non-coal class includes shale, sandstone, limestone, siltstone, and clay.", 0.68, 5.35, 7.2, 0.4, size=15, color=MUTED)
    add_footer(slide, 2)
    add_notes(slide, "The objective is simple to state: classify coal versus non-coal. The difficulty comes from the geological diversity inside the non-coal class and from the need to generalize to unseen material types.")

    # 3. Why it matters
    slide = prs.slides.add_slide(blank)
    set_background(slide, WHITE)
    add_eyebrow(slide, "02 / THE CONTEXT")
    add_heading(slide, "The visual difference is not always enough.")
    add_text(slide, "Coal sits inside a family of visually similar geological materials. Spectroscopy adds a measurable chemical and structural signal to the decision.", 0.68, 2.0, 6.6, 0.85, size=20, color=MUTED)
    add_image(slide, FIGURES / "coal_noncoal_problem_materials.png", 7.4, 1.35, 5.2, 4.9)
    add_text(slide, "A foundation for automated coal identification using spectroscopy.", 0.68, 4.35, 5.4, 0.45, size=20, color=FOREST, bold=True)
    add_text(slide, "Current scope: NIR spectra. Future scope: hyperspectral image cubes.", 0.68, 5.0, 5.5, 0.4, size=12, color=MUTED, font=MONO)
    add_footer(slide, 3)
    add_notes(slide, "Keep the distinction clear: this project uses one-dimensional NIR spectra, not pixel-level hyperspectral images. Hyperspectral imaging is a future extension, not the current input.")

    # 4. Materials
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_eyebrow(slide, "03 / THE MATERIALS")
    add_heading(slide, "One binary decision. Twenty-four material groups.")
    add_text(slide, "The model sees two labels, but the experiment preserves the geological identity behind every spectrum.", 0.68, 1.9, 7.5, 0.45, size=19, color=MUTED)
    coal_names = "0.3 coking coal\nAnthracite grades 1–2\nCoking, fat, gas, and gas-fat coal\nLean, lean-thin, lignite grades 1–2\nThin coal"
    rock_names = "Arenaceous shale\nArgillaceous limestone\nBlack shale\nCarbonaceous shale\nClay\nFine-grained sandstone\nSiltstone"
    add_box(slide, 0.68, 2.8, 5.7, 3.0, fill=FOREST, line=FOREST)
    add_text(slide, "COAL / 12 GROUPS", 1.0, 3.1, 3.3, 0.25, size=10, color=AMBER, font=MONO)
    add_text(slide, coal_names, 1.0, 3.55, 4.8, 1.8, size=17, color=WHITE)
    add_box(slide, 6.9, 2.8, 5.7, 3.0, fill=WHITE, line=LINE)
    add_text(slide, "NON-COAL / 12 GROUPS", 7.22, 3.1, 3.6, 0.25, size=10, color=CORAL, font=MONO)
    add_text(slide, rock_names, 7.22, 3.55, 4.8, 1.8, size=17, color=INK)
    add_footer(slide, 4)
    add_notes(slide, "The 24 material groups are central to the validation design. The classifier is not allowed to gain an advantage by seeing spectra from the same material in both sides of a split.")

    # 5. Spectrum
    slide = prs.slides.add_slide(blank)
    set_background(slide, WHITE)
    add_eyebrow(slide, "04 / THE INPUT")
    add_heading(slide, "What is an NIR spectrum?")
    add_text(slide, "Each sample becomes a one-dimensional fingerprint: intensity measured across 1,500 wavelengths.", 0.68, 1.9, 7.3, 0.5, size=19, color=MUTED)
    add_image(slide, FIGURES / "lignite_vs_argillaceous_limestone_raw.png", 0.68, 2.75, 8.0, 3.8)
    add_box(slide, 9.15, 2.75, 3.45, 3.8, fill=PAPER, line=LINE)
    add_text(slide, "SPECTRAL WINDOW", 9.5, 3.15, 2.6, 0.2, size=9, color=CORAL, font=MONO)
    add_text(slide, "1000–2499", 9.5, 3.7, 2.7, 0.55, size=31, color=FOREST, bold=True)
    add_text(slide, "nanometres", 9.5, 4.35, 2.4, 0.25, size=11, color=MUTED, font=MONO)
    add_text(slide, "Coal and limestone carry different curves across the same measured range.", 9.5, 5.05, 2.55, 0.7, size=15, color=INK)
    add_footer(slide, 5)
    add_notes(slide, "Use the plot to explain the data representation. This is not an image of the sample; it is the measured response of the sample across wavelengths.")

    # 6. Dataset
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_eyebrow(slide, "05 / THE DATASET")
    add_heading(slide, "A broad measurement set, with a clear grain.")
    add_text(slide, "The original CSV measurements were converted into arrays and metadata that can be used consistently by every experiment.", 0.68, 1.88, 8.2, 0.5, size=18, color=MUTED)
    add_metric_card(slide, "10,496", "measured spectra", 0.68, 3.0)
    add_metric_card(slide, "1,500", "features / sample", 3.62, 3.0)
    add_metric_card(slide, "5,016", "coal spectra", 6.56, 3.0)
    add_metric_card(slide, "5,480", "non-coal spectra", 9.5, 3.0)
    add_box(slide, 0.68, 4.72, 11.86, 0.8, fill=FOREST, line=FOREST)
    add_text(slide, "X.npy  /  y.npy  /  wavelengths.npy  /  metadata.csv", 1.0, 4.99, 11.2, 0.25, size=14, color=MINT, font=MONO, align=PP_ALIGN.CENTER)
    add_footer(slide, 6)
    add_notes(slide, "Explain the four core files only at a high level. The audience does not need to know implementation details, but should understand that spectra, labels, wavelengths, and material metadata are kept together.")

    # 7. Categories
    slide = prs.slides.add_slide(blank)
    set_background(slide, WHITE)
    add_eyebrow(slide, "06 / THE LABELS")
    add_heading(slide, "The class balance is close, but the geology is not.")
    add_text(slide, "Coal and non-coal are broadly balanced in the full dataset. The harder problem is variation between material groups.", 0.68, 1.9, 8.1, 0.5, size=19, color=MUTED)
    add_text(slide, "FULL DATASET", 0.68, 3.0, 2.2, 0.22, size=9, color=CORAL, font=MONO)
    add_box(slide, 0.68, 3.45, 11.86, 0.46, fill=MINT, line=MINT)
    add_box(slide, 0.68, 3.45, 5.67, 0.46, fill=FOREST, line=FOREST)
    add_text(slide, "COAL  5,016 / 47.8%", 1.0, 3.57, 3.0, 0.2, size=10, color=WHITE, font=MONO)
    add_text(slide, "NON-COAL  5,480 / 52.2%", 7.55, 3.57, 3.8, 0.2, size=10, color=FOREST, font=MONO)
    add_text(slide, "What changes across groups:", 0.68, 4.55, 3.2, 0.25, size=16, color=INK, bold=True)
    add_bullet_list(slide, ["Particle size and measurement conditions", "Chemical composition and geological origin", "Spectral shape, offset, and local absorption features"], 0.68, 5.0, 8.0, line_height=0.42, size=14, color=MUTED)
    add_footer(slide, 7)
    add_notes(slide, "This slide helps the audience understand why a random sample split would be misleading. A similar-looking set of spectra from one material can dominate a test set if group identity is ignored.")

    # 8. Example measurements
    slide = prs.slides.add_slide(blank)
    set_background(slide, PAPER)
    add_eyebrow(slide, "07 / THE EVIDENCE")
    add_heading(slide, "The signal is measurable before the model sees it.")
    add_image(slide, FIGURES / "lignite_vs_argillaceous_limestone_raw.png", 0.68, 1.9, 8.1, 4.6)
    add_box(slide, 9.25, 2.15, 3.3, 3.9, fill=WHITE, line=LINE)
    add_text(slide, "READING THE PLOT", 9.58, 2.55, 2.4, 0.2, size=9, color=CORAL, font=MONO)
    add_text(slide, "Two materials can be compared on the same wavelength axis.", 9.58, 3.05, 2.45, 0.8, size=20, color=INK, bold=True)
    add_text(slide, "The classifier turns this shape information into a repeatable coal / non-coal decision.", 9.58, 4.35, 2.45, 0.8, size=13, color=MUTED)
    add_footer(slide, 8)
    add_notes(slide, "This is a genuine project-generated comparison figure. Avoid claiming that visual separation alone proves classification quality; the validation study is what tests generalization.")

    # 9. Workflow
    slide = prs.slides.add_slide(blank)
    set_background(slide, INK)
    add_eyebrow(slide, "08 / THE WORKFLOW", dark=True)
    add_heading(slide, "A measured signal becomes a controlled decision.", color=WHITE)
    steps = [("01", "Prepare", "CSV -> arrays"), ("02", "Split", "by material"), ("03", "Transform", "scale + PCA"), ("04", "Compare", "grouped CV"), ("05", "Predict", "coal / non-coal")]
    x = 0.68
    for index, (num, title, sub) in enumerate(steps):
        add_box(slide, x, 2.65, 2.15, 1.65, fill=FOREST if index != 4 else CORAL, line=FOREST if index != 4 else CORAL)
        add_text(slide, num, x + 0.2, 2.9, 0.5, 0.2, size=9, color=AMBER if index != 4 else WHITE, font=MONO)
        add_text(slide, title, x + 0.2, 3.38, 1.8, 0.3, size=18, color=WHITE, bold=True)
        add_text(slide, sub, x + 0.2, 3.83, 1.8, 0.2, size=9, color=MINT if index != 4 else WHITE, font=MONO)
        if index < 4:
            add_text(slide, "→", x + 2.25, 3.3, 0.35, 0.3, size=20, color=AMBER, align=PP_ALIGN.CENTER)
        x += 2.52
    add_text(slide, "Every transformation is fitted using training data only.", 0.68, 5.35, 7.5, 0.25, size=13, color=RGBColor(210, 218, 210), font=MONO)
    add_footer(slide, 9, dark=True)
    add_notes(slide, "Walk through the project as a sequence of controlled decisions. The central discipline is that transformations are not allowed to learn from validation or holdout data.")

    # 10. Preparation
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_eyebrow(slide, "09 / DATA PREPARATION")
    add_heading(slide, "From CSV measurements to a usable table.")
    add_pipeline_step(slide, "01", "Raw CSV", "metadata + measurements", 0.68, 2.45, 2.65)
    add_text(slide, "→", 3.45, 2.93, 0.45, 0.3, size=22, color=CORAL, align=PP_ALIGN.CENTER)
    add_pipeline_step(slide, "02", "Clean rows", "valid wavelengths", 4.05, 2.45, 2.65)
    add_text(slide, "→", 6.82, 2.93, 0.45, 0.3, size=22, color=CORAL, align=PP_ALIGN.CENTER)
    add_pipeline_step(slide, "03", "Assign labels", "material metadata", 7.42, 2.45, 2.65)
    add_text(slide, "→", 10.2, 2.93, 0.45, 0.3, size=22, color=CORAL, align=PP_ALIGN.CENTER)
    add_pipeline_step(slide, "04", "Save arrays", "X / y / wavelengths", 10.8, 2.45, 1.75, fill=CORAL, text_color=WHITE)
    add_text(slide, "The same wavelength coordinate is preserved across every sample, making model comparisons meaningful.", 0.68, 4.75, 8.2, 0.5, size=20, color=FOREST, bold=True)
    add_text(slide, "Invalid spectra are removed before the modeling pipeline begins.", 0.68, 5.55, 7.4, 0.3, size=13, color=MUTED, font=MONO)
    add_footer(slide, 10)
    add_notes(slide, "Keep this slide conceptual. Mention that the raw CSV files contain metadata and spectral measurements, which are validated and converted into arrays plus material metadata.")

    # 11. Validation
    slide = prs.slides.add_slide(blank)
    set_background(slide, WHITE)
    add_eyebrow(slide, "10 / MATERIAL-AWARE VALIDATION")
    add_heading(slide, "The test asks a harder question: have we seen this material before?")
    add_text(slide, "The development and holdout sets share zero material groups.", 0.68, 1.9, 7.0, 0.4, size=20, color=MUTED)
    add_box(slide, 0.68, 2.8, 5.3, 2.55, fill=PAPER, line=LINE)
    add_text(slide, "DEVELOPMENT", 1.0, 3.15, 2.2, 0.2, size=9, color=CORAL, font=MONO)
    add_text(slide, "8,142", 1.0, 3.55, 2.2, 0.6, size=38, color=FOREST, bold=True)
    add_text(slide, "19 materials", 1.0, 4.28, 2.0, 0.25, size=11, color=MUTED, font=MONO)
    add_box(slide, 6.58, 2.8, 5.3, 2.55, fill=FOREST, line=FOREST)
    add_text(slide, "FINAL HOLDOUT", 6.9, 3.15, 2.5, 0.2, size=9, color=AMBER, font=MONO)
    add_text(slide, "2,354", 6.9, 3.55, 2.2, 0.6, size=38, color=WHITE, bold=True)
    add_text(slide, "5 unseen materials", 6.9, 4.28, 2.8, 0.25, size=11, color=MINT, font=MONO)
    add_text(slide, "No material overlap", 4.65, 5.75, 4.0, 0.25, size=12, color=CORAL, font=MONO, align=PP_ALIGN.CENTER)
    add_footer(slide, 11)
    add_notes(slide, "Explain material grouping in everyday terms: all measurements from one named material stay together. This keeps the evaluation focused on generalization, not recognition of near-duplicates.")

    # 12. Models
    slide = prs.slides.add_slide(blank)
    set_background(slide, PAPER)
    add_eyebrow(slide, "11 / MODELS COMPARED")
    add_heading(slide, "Five ways to make the decision.")
    models = [("SVM", "stable primary", FOREST), ("XGBoost", "close alternative", MINT), ("PLS-DA", "high recall", AMBER), ("Random Forest", "tree benchmark", LINE), ("1D CNN", "current baseline", CORAL)]
    y = 2.1
    for index, (name, role, fill) in enumerate(models):
        add_box(slide, 0.68, y, 11.86, 0.7, fill=WHITE if index != 0 else FOREST, line=LINE if index != 0 else FOREST)
        add_text(slide, f"0{index + 1}", 0.95, y + 0.24, 0.4, 0.18, size=9, color=AMBER if index == 0 else CORAL, font=MONO)
        add_text(slide, name, 1.65, y + 0.18, 3.2, 0.3, size=19, color=WHITE if index == 0 else INK, bold=True)
        add_text(slide, role.upper(), 9.0, y + 0.25, 2.8, 0.2, size=9, color=MINT if index == 0 else MUTED, font=MONO, align=PP_ALIGN.RIGHT)
        y += 0.86
    add_footer(slide, 12)
    add_notes(slide, "Introduce each model by role rather than implementation detail. SVM is the selected stable model; the others show how different modeling assumptions behaved under the same grouped validation protocol.")

    # 13. Selected pipeline
    slide = prs.slides.add_slide(blank)
    set_background(slide, INK)
    add_eyebrow(slide, "12 / SELECTED PIPELINE", dark=True)
    add_heading(slide, "A compact model with a disciplined handoff.", color=WHITE)
    add_pipeline_step(slide, "01", "Raw spectrum", "1,500 values", 0.68, 2.45, 2.65, fill=FOREST, text_color=WHITE)
    add_text(slide, "→", 3.45, 2.93, 0.45, 0.3, size=22, color=AMBER, align=PP_ALIGN.CENTER)
    add_pipeline_step(slide, "02", "StandardScaler", "training-only fit", 4.05, 2.45, 2.65, fill=FOREST, text_color=WHITE)
    add_text(slide, "→", 6.82, 2.93, 0.45, 0.3, size=22, color=AMBER, align=PP_ALIGN.CENTER)
    add_pipeline_step(slide, "03", "PCA / 2D", "99.4720% variance", 7.42, 2.45, 2.65, fill=FOREST, text_color=WHITE)
    add_text(slide, "→", 10.2, 2.93, 0.45, 0.3, size=22, color=AMBER, align=PP_ALIGN.CENTER)
    add_pipeline_step(slide, "04", "RBF SVM", "C=1, gamma=scale", 10.8, 2.45, 1.75, fill=CORAL, text_color=WHITE)
    add_text(slide, "The final decision is binary. The validation design behind it is not.", 0.68, 5.1, 7.6, 0.45, size=22, color=MINT, bold=True)
    add_footer(slide, 13, dark=True)
    add_notes(slide, "This is the pipeline to feature in the presentation title and conclusion. Explain that PCA creates two learned combinations of the original wavelengths; it does not select two individual wavelengths.")

    # 14. Repeated results
    slide = prs.slides.add_slide(blank)
    set_background(slide, WHITE)
    add_eyebrow(slide, "13 / REPEATED VALIDATION")
    add_heading(slide, "SVM stays steady when the material changes.")
    add_text(slide, "Mean results across 3 seeds and 5 material-grouped folds.", 0.68, 1.85, 7.0, 0.3, size=15, color=MUTED, font=MONO)
    add_text(slide, "MEAN ACCURACY", 1.95, 2.65, 2.3, 0.2, size=9, color=MUTED, font=MONO)
    rows = [("SVM", 93.60, FOREST), ("XGBoost", 93.25, RGBColor(90, 130, 110)), ("Random Forest", 92.84, RGBColor(160, 173, 155)), ("PLS-DA", 82.11, AMBER)]
    y = 3.15
    for name, value, color in rows:
        add_text(slide, name, 0.68, y - 0.03, 1.1, 0.2, size=11, color=INK, bold=name == "SVM")
        add_box(slide, 1.95, y, 8.2, 0.32, fill=LINE, line=LINE)
        add_box(slide, 1.95, y, 8.2 * value / 100, 0.32, fill=color, line=color)
        add_text(slide, f"{value:.2f}%", 10.45, y - 0.03, 1.1, 0.2, size=11, color=INK, font=MONO, align=PP_ALIGN.RIGHT)
        y += 0.72
    add_box(slide, 0.68, 6.18, 11.86, 0.45, fill=PAPER, line=PAPER)
    add_text(slide, "SVM: strongest combined accuracy + F1 stability.  PLS-DA: highest recall, but much more variable.", 0.92, 6.31, 11.3, 0.18, size=10, color=FOREST, font=MONO, align=PP_ALIGN.CENTER)
    add_footer(slide, 14)
    add_notes(slide, "This is the primary model-selection result. The chart uses the actual repeated grouped-validation values. Avoid treating the single holdout score as the only evidence.")

    # 15. Holdout comparison
    slide = prs.slides.add_slide(blank)
    set_background(slide, PAPER)
    add_eyebrow(slide, "14 / HOLDOUT COMPARISON")
    add_heading(slide, "A useful snapshot, with an important footnote.")
    add_text(slide, "These are the stored results from the material-disjoint holdout comparison. Multiple candidates were evaluated on it later, so it is not a pristine final benchmark.", 0.68, 1.85, 10.3, 0.55, size=15, color=MUTED)
    headers = ["Model", "Accuracy", "Coal precision", "Coal recall", "F1"]
    x_positions = [0.68, 4.55, 6.25, 8.35, 10.35]
    for header, x in zip(headers, x_positions):
        add_text(slide, header.upper(), x, 2.75, 1.7, 0.2, size=8, color=MUTED, font=MONO)
    holdout_rows = [("SVM", "95.84%", "99.07%", "89.11%", "93.83%"), ("PLS-DA", "96.90%", "93.01%", "98.68%", "95.76%"), ("XGBoost", "96.47%", "98.33%", "91.63%", "94.86%"), ("Random Forest", "93.54%", "98.58%", "83.01%", "90.13%"), ("CNN", "62.49%", "39.27%", "10.29%", "16.30%")]
    y = 3.22
    for row_index, row in enumerate(holdout_rows):
        if row_index == 0:
            add_box(slide, 0.55, y - 0.08, 11.95, 0.54, fill=WHITE, line=WHITE)
        for value, x in zip(row, x_positions):
            add_text(slide, value, x, y, 1.7, 0.2, size=11, color=CORAL if row_index == 0 and x == 0.68 else INK, bold=row_index == 0 and x == 0.68, font=FONT if x == 0.68 else MONO)
        y += 0.58
    add_footer(slide, 15)
    add_notes(slide, "Explain that the table is useful for understanding the trade-off between precision and recall, but the repeated grouped-validation summary is stronger for selecting the stable model.")

    # 16. Confusion matrix
    slide = prs.slides.add_slide(blank)
    set_background(slide, CORAL)
    add_eyebrow(slide, "15 / ERROR PROFILE", dark=True)
    add_heading(slide, "What the selected SVM gets right and wrong.", color=WHITE)
    add_text(slide, "The model is precise when it says coal, but it still misses some actual coal spectra.", 0.68, 1.82, 7.4, 0.4, size=18, color=WHITE)
    add_text(slide, "PREDICTED", 7.35, 2.58, 2.7, 0.2, size=9, color=WHITE, font=MONO, align=PP_ALIGN.CENTER)
    add_text(slide, "ACTUAL", 6.42, 4.33, 0.9, 0.2, size=9, color=WHITE, font=MONO, align=PP_ALIGN.CENTER)
    add_text(slide, "Non-Coal", 7.2, 3.05, 1.5, 0.2, size=10, color=WHITE, font=MONO, align=PP_ALIGN.CENTER)
    add_text(slide, "Coal", 9.45, 3.05, 1.0, 0.2, size=10, color=WHITE, font=MONO, align=PP_ALIGN.CENTER)
    add_text(slide, "Non-Coal", 5.2, 3.85, 1.5, 0.2, size=10, color=WHITE, font=MONO, align=PP_ALIGN.RIGHT)
    add_text(slide, "Coal", 5.7, 5.0, 1.0, 0.2, size=10, color=WHITE, font=MONO, align=PP_ALIGN.RIGHT)
    add_box(slide, 6.9, 3.45, 2.2, 1.25, fill=WHITE, line=WHITE)
    add_box(slide, 9.1, 3.45, 2.2, 1.25, fill=RGBColor(120, 55, 43), line=RGBColor(120, 55, 43))
    add_box(slide, 6.9, 4.7, 2.2, 1.25, fill=RGBColor(120, 55, 43), line=RGBColor(120, 55, 43))
    add_box(slide, 9.1, 4.7, 2.2, 1.25, fill=WHITE, line=WHITE)
    add_text(slide, "1,511", 6.9, 3.82, 2.2, 0.35, size=29, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "7", 9.1, 3.82, 2.2, 0.35, size=29, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "91", 6.9, 5.07, 2.2, 0.35, size=29, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "745", 9.1, 5.07, 2.2, 0.35, size=29, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "true negatives", 6.9, 4.42, 2.2, 0.16, size=7, color=MUTED, font=MONO, align=PP_ALIGN.CENTER)
    add_text(slide, "false positives", 9.1, 4.42, 2.2, 0.16, size=7, color=WHITE, font=MONO, align=PP_ALIGN.CENTER)
    add_text(slide, "false negatives", 6.9, 5.77, 2.2, 0.16, size=7, color=WHITE, font=MONO, align=PP_ALIGN.CENTER)
    add_text(slide, "true positives", 9.1, 5.77, 2.2, 0.16, size=7, color=MUTED, font=MONO, align=PP_ALIGN.CENTER)
    add_footer(slide, 16, dark=True)
    add_notes(slide, "Define false negatives in plain language: actual coal that the model calls non-coal. This is the main remaining error type in the SVM holdout snapshot.")

    # 17. Difficult materials
    slide = prs.slides.add_slide(blank)
    set_background(slide, WHITE)
    add_eyebrow(slide, "16 / LIMITATIONS")
    add_heading(slide, "The difficult materials are the real story.")
    add_text(slide, "Material-level performance shows where generalization becomes difficult.", 0.68, 1.88, 7.0, 0.35, size=18, color=MUTED)
    add_image(slide, FIGURES / "difficult_materials_comparison.png", 0.68, 2.65, 7.3, 3.8)
    add_box(slide, 8.45, 2.65, 4.1, 3.8, fill=PAPER, line=LINE)
    add_text(slide, "RECURRING HARD GROUPS", 8.8, 3.02, 3.0, 0.2, size=9, color=CORAL, font=MONO)
    add_bullet_list(slide, ["Lignite coal grade 2", "Carbonaceous shale", "Black shale 1", "Siltstone 2", "Arenaceous shale 1"], 8.8, 3.5, 3.1, line_height=0.43, size=14, color=INK)
    add_text(slide, "The lesson is not simply model complexity. It is the breadth of geological variation represented in the data.", 8.8, 5.72, 3.05, 0.48, size=11, color=MUTED)
    add_footer(slide, 17)
    add_notes(slide, "Use this slide to make the limitations concrete. The difficult groups recur across model comparisons, suggesting a data-domain challenge rather than a single algorithm defect.")

    # 18. Conclusion
    slide = prs.slides.add_slide(blank)
    set_background(slide, FOREST)
    add_eyebrow(slide, "17 / CONCLUSION", dark=True)
    add_heading(slide, "A careful split made the result more useful.", color=WHITE)
    add_bullet_list(slide, ["SVM is the most stable primary pipeline.", "XGBoost is a close alternative with slightly higher mean recall.", "PLS-DA is useful when coal recall is the highest priority.", "Random Forest and the current CNN are weaker in these experiments.", "Independent external validation is still required."], 0.78, 2.25, 7.2, line_height=0.58, size=18, color=WHITE)
    add_box(slide, 9.15, 2.2, 3.25, 2.65, fill=CORAL, line=CORAL)
    add_text(slide, "NEXT", 9.5, 2.58, 2.4, 0.2, size=10, color=AMBER, font=MONO)
    add_text(slide, "Broader spectra.\nIndependent materials.\nThen hyperspectral images.", 9.5, 3.15, 2.45, 1.1, size=20, color=WHITE, bold=True)
    add_text(slide, "NIR spectra today / image cubes tomorrow", 0.78, 6.25, 6.2, 0.2, size=10, color=MINT, font=MONO)
    add_footer(slide, 18, dark=True)
    add_notes(slide, "Close with the balanced claim: the study demonstrates a promising, carefully validated pipeline, while the difficult material groups and lack of an independent external dataset limit stronger claims.")

    # 19. References
    slide = prs.slides.add_slide(blank)
    set_background(slide, PAPER)
    add_eyebrow(slide, "18 / REFERENCES")
    add_heading(slide, "Sources and further reading.")
    refs = [
        ("01", "Dataset", "Lv, Y., Wang, S., Yang, E., & Ge, S. (2024). A near infrared spectroscopy dataset of coal and coal-measure rock under diverse conditions. Zenodo.", "https://doi.org/10.5281/zenodo.11137126"),
        ("02", "Project", "Coal Detection project repository and reproducible code.", "https://github.com/spacewhale-zion/coal-detection"),
        ("03", "Software", "Scikit-learn documentation: PCA, SVM, metrics, and grouped validation.", "https://scikit-learn.org/"),
        ("04", "Software", "XGBoost documentation: gradient-boosted decision trees.", "https://xgboost.readthedocs.io/"),
        ("05", "Method", "RockSL and public spectral-library resources were considered as external geological context, not merged into training.", "https://github.com/CSU-PCP-XBS/spectral-dataset-RockSL"),
    ]
    y = 2.05
    for number, category, text, link in refs:
        add_text(slide, number, 0.68, y, 0.35, 0.2, size=9, color=CORAL, font=MONO)
        add_text(slide, category.upper(), 1.28, y, 1.15, 0.2, size=8, color=MUTED, font=MONO)
        add_text(slide, text, 2.75, y - 0.03, 7.15, 0.4, size=12, color=INK)
        add_text(slide, link, 2.75, y + 0.38, 8.2, 0.2, size=9, color=FOREST, font=MONO)
        add_box(slide, 0.68, y + 0.75, 11.8, 0.01, fill=LINE, line=LINE)
        y += 0.93
    add_text(slide, "All figures and metrics in this presentation were generated by the project or sourced from the references above.", 0.68, 6.45, 10.5, 0.25, size=10, color=MUTED, font=MONO)
    add_footer(slide, 19)
    add_notes(slide, "Use this final slide for source attribution. The dataset source, project repository, software documentation, and spectral-library context are separated clearly.")

    prs.save(OUTPUT)
    print(f"Saved presentation to: {OUTPUT}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    make_deck()