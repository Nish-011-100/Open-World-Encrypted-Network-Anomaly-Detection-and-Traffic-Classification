"""Build the concise NetAnomaly-OW research review PDF."""

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
TMP = ROOT / "tmp" / "pdfs"
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, TEAL, PALE = "#12263A", "#2563EB", "#0F9D8A", "#EFF6FF"


def chart_comparison():
    names = ["Ensemble", "Transformer", "xLSTM", "DriftMamba"]
    acc = [0.823, 0.806, 0.799, 0.755]
    f1 = [0.7264, 0.6981, 0.6901, 0.6435]
    auroc = [0.6433, 0.6090, 0.5643, 0.7011]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    x = range(len(names)); w = .24
    ax.bar([i-w for i in x], acc, w, label="Accuracy", color=BLUE)
    ax.bar(x, f1, w, label="Macro-F1", color=TEAL)
    ax.bar([i+w for i in x], auroc, w, label="Unknown AUROC", color="#F59E0B")
    ax.set_ylim(0.5, 0.86); ax.set_xticks(list(x), names); ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=3, loc="upper center", frameon=False); fig.tight_layout()
    path = TMP / "model_comparison.png"; fig.savefig(path, dpi=190, transparent=False); plt.close(fig)
    return path


def chart_flags():
    names = ["DriftMamba", "Transformer", "xLSTM", "Ensemble"]
    known = [110, 88, 92, 51]; unknown = [292, 190, 145, 116]
    fig, ax = plt.subplots(figsize=(8.2, 2.0))
    ax.bar(names, known, label="Known flows rejected", color="#EF4444")
    ax.bar(names, unknown, bottom=known, label="Held-out unknown detected", color=TEAL)
    ax.set_ylabel("Flagged flows (of 2,000)"); ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=2, loc="upper right"); fig.tight_layout()
    path = TMP / "flagging.png"; fig.savefig(path, dpi=190); plt.close(fig)
    return path


def chart_architecture():
    """Render the end-to-end NetAnomaly-OW model architecture."""
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, width, height, title, detail, color, text_color="white"):
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.4, edgecolor=color, facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x, y + height * .27, title, ha="center", va="center", color=text_color,
                fontsize=9.6, fontweight="bold", linespacing=1.0)
        ax.text(x, y - height * .10, detail, ha="center", va="center", color=text_color,
                fontsize=7.8, linespacing=1.15)

    def arrow(start, end, color="#64748B"):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
            color=color, shrinkA=3, shrinkB=3,
        ))

    stages = [
        (.10, .16, "1\nEncrypted flows",
         "CESNET / Wireshark\nBidirectional flows\nRedacted endpoints", NAVY, "white"),
        (.29, .18, "2\nMulti-view features",
         "Packet tensor [B, 64, 3]\nValidity mask [B, 64]\nAggregates + signatures",
         TEAL, "white"),
        (.49, .18, "3\nEncoder alternatives",
         "DriftMamba\nCausal Transformer\nxLSTM-inspired", "#334155", "white"),
        (.70, .20, "4\nOpen-world head",
         "Fusion + 64-D embedding\nTwo prototypes per class\nConformal p-value\nOptional ensemble",
         BLUE, "white"),
        (.90, .16, "5\nDecision evidence",
         "Application label\nKNOWN / UNKNOWN\nPrediction set\nSimilarity + uncertainty",
         NAVY, "white"),
    ]
    for x, width, title, detail, color, text_color in stages:
        box(x, .60, width, .46, title, detail, color, text_color)

    for left, right in pairwise(stages):
        start_x = left[0] + left[1] / 2
        end_x = right[0] - right[1] / 2
        arrow((start_x, .60), (end_x, .60))

    box(.50, .16, .74, .14, "Continuous evaluation and drift audit",
        "8 / 16 / 32 / 64 packet prefixes   |   feature PSI   |   embedding-centroid shift",
        "#E2E8F0", NAVY)
    arrow((.90, .36), (.87, .23), "#0F9D8A")

    fig.tight_layout(pad=.35)
    path = TMP / "system_architecture.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def footer(canvas, doc):
    canvas.setAuthor("Nish-011-100")
    canvas.setTitle("Open-World Encrypted Network Anomaly Detection and Traffic Classification")
    canvas.setSubject("NetAnomaly-OW research review and experimental report")
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(18*mm, 14*mm, 192*mm, 14*mm); canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18*mm, 9*mm, "NetAnomaly-OW | concise research review")
    canvas.drawRightString(192*mm, 9*mm, f"Page {doc.page}"); canvas.restoreState()


def build():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=22, leading=25, textColor=colors.HexColor(NAVY), spaceAfter=6))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=9.5, leading=14,
                              textColor=colors.HexColor("#475569"), spaceAfter=10))
    styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontSize=14, leading=17,
                              textColor=colors.HexColor(NAVY), spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=8.7, leading=12.2,
                              textColor=colors.HexColor("#243447"), spaceAfter=5))
    styles.add(ParagraphStyle(name="Cap", parent=styles["Normal"], fontSize=7.5, leading=10,
                              alignment=TA_CENTER, textColor=colors.HexColor("#64748B")))
    styles.add(ParagraphStyle(name="Cell", parent=styles["Normal"], fontSize=7.0, leading=8.8,
                              textColor=colors.HexColor("#111827")))
    styles.add(ParagraphStyle(name="CellHead", parent=styles["Cell"], fontName="Helvetica-Bold",
                              textColor=colors.white))
    doc = BaseDocTemplate(str(OUT / "NetAnomaly_OW_Research_Review.pdf"), pagesize=A4,
                          rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=18*mm)
    doc.addPageTemplates(PageTemplate(id="main", frames=Frame(18*mm, 18*mm, 174*mm, 263*mm,
                                                               id="body"), onPage=footer))
    story = [Paragraph(
        "Open-World Encrypted Network Anomaly Detection and Traffic Classification",
        styles["TitleX"],
    ), Paragraph(
        "NetAnomaly-OW | selective-state, attention, stabilized recurrent, and ensemble inference",
        styles["Sub"],
    )]
    callout = Table([[Paragraph("Research question", styles["BodyX"]), Paragraph(
        "Can encrypted applications be classified early while rejecting applications absent from training?", styles["BodyX"])],
        [Paragraph("Evidence", styles["BodyX"]), Paragraph(
        "Official CESNET-QUIC22 XS; 5,000 train, 1,000 calibration, 1,000 known-test and 1,000 held-out-unknown flows; chronological week-to-week split.", styles["BodyX"])]], colWidths=[31*mm, 139*mm])
    callout.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor(PALE)),("BOX",(0,0),(-1,-1),.6,colors.HexColor("#93C5FD")),("INNERGRID",(0,0),(-1,-1),.3,colors.white),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
    story += [callout, Paragraph("Abstract", styles["H1X"]), Paragraph(
        "This study investigates encrypted QUIC application recognition under an open-world setting, where test traffic may belong to services absent from training. The proposed system combines flow-level behavioral context, packet-sequence encoders, multi-prototype classification, conformal unknown rejection, early-flow evaluation, and chronological drift monitoring. The objective is not payload inspection or attack attribution, but reliable application classification and novelty triage from encrypted metadata.", styles["BodyX"]),
        Paragraph("1. Research Objective and Technical Contribution", styles["H1X"]), Paragraph(
        "Each bidirectional flow is represented by signed packet sizes, directions, inter-arrival times, aggregate statistics, and second-order path signatures. DriftMamba, a causal Transformer, and an xLSTM-inspired encoder share a hyperspherical multi-prototype head. Their heterogeneous ensemble combines complementary state-space, attention, and recurrent evidence. Split conformal calibration converts model evidence into known-traffic p-values and prediction sets; chronological windows provide drift auditing.", styles["BodyX"])]
    arch = Table([["Packets + timing", "Encoder", "Context fusion", "Prototype head", "Conformal decision"],
                  ["up to 64 positions", "Mamba / Transformer / xLSTM", "aggregate + signatures", "application similarity", "KNOWN / UNKNOWN"]], colWidths=[32*mm,40*mm,36*mm,32*mm,30*mm])
    arch.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("BACKGROUND",(0,1),(-1,1),colors.HexColor("#F8FAFC")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#CBD5E1")),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.3),("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    protocol = Table([
        ["Stage", "Rows", "Purpose", "Temporal scope"],
        ["Training", "5,000", "Learn encoders and prototypes", "CESNET week 44"],
        ["Calibration", "1,000", "Fit conformal rejection", "CESNET week 44"],
        ["Known test", "1,000", "15 trained applications", "CESNET week 45"],
        ["Unknown test", "1,000", "58 held-out applications", "CESNET week 45"],
    ], colWidths=[32*mm,22*mm,72*mm,44*mm], repeatRows=1)
    protocol.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(TEAL)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("FONTSIZE",(0,0),(-1,-1),7.6),("ALIGN",(1,1),(1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story += [arch, Paragraph("2. Experimental Design and Leakage Controls", styles["H1X"]), Paragraph(
        "The chronological split is applied before feature fitting. Preprocessors, labels, anomaly thresholds, and conformal scores are learned only from training or calibration partitions. Testing uses the following week, preventing future-flow leakage and making unknown applications operationally meaningful.", styles["BodyX"]), protocol, PageBreak(),
        Paragraph("3. End-to-End Processing Pipeline", styles["H1X"]), Paragraph(
        "The system accepts official CESNET DataZoo frames or Wireshark-style packet exports. It constructs privacy-preserving bidirectional flows, derives synchronized sequence and statistical views, trains causal encoders, calibrates unknown rejection on a disjoint partition, and emits application, novelty, uncertainty, and drift outputs.", styles["BodyX"])]
    pipeline = Table([
        ["Stage", "Input", "Operation", "Output"],
        ["1. Ingestion", "Packet/flow records", "Schema validation and timestamp parsing", "Canonical records"],
        ["2. Flow building", "Endpoints and protocol", "Bidirectional grouping; endpoint redaction", "Flow table"],
        ["3. Representation", "First packet positions", "Sequence tensor, mask, aggregates, signatures", "Multi-view tensors"],
        ["4. Temporal split", "Ordered flows", "Train/calibration/known/unknown partitioning", "Leakage-free subsets"],
        ["5. Learning", "Training subset", "Encoder and prototype optimization", "Neural checkpoint"],
        ["6. Calibration", "Known calibration subset", "Split-conformal score fitting", "P-value calibrator"],
        ["7. Operation", "New encrypted flow", "Classify, reject, audit drift", "Decision and evidence"],
    ], colWidths=[29*mm,35*mm,72*mm,34*mm], repeatRows=1)
    pipeline.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("FONTSIZE",(0,0),(-1,-1),7.25),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    features = Table([
        ["Feature view", "Representative variables", "Purpose"],
        ["Packet sequence", "Signed size, direction, inter-arrival time, validity mask", "Preserve order and request-response dynamics"],
        ["Aggregate context", "Duration, bytes, packet counts, ratios and distribution summaries", "Describe complete flow behavior"],
        ["Path signature", "Second-order interactions over size, direction and time", "Encode trajectory shape beyond simple moments"],
        ["Operational context", "Protocol/ports and chronological start time where available", "Support routing, splitting and drift windows"],
    ], colWidths=[34*mm,82*mm,54*mm], repeatRows=1)
    features.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(TEAL)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("FONTSIZE",(0,0),(-1,-1),7.25),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    models = Table([
        ["Pipeline", "Sequence mechanism", "Role and trade-off"],
        ["DriftMamba", "Input-selective causal state updates; linear scan", "Strongest neural unknown separation; streaming-friendly"],
        ["Causal Transformer", "Masked multi-head self-attention with learned positions", "Highest individual accuracy; quadratic attention cost"],
        ["xLSTM-inspired", "Stabilized exponential gates and normalized scalar memory", "Strong early-packet behavior; weaker unknown separation"],
        ["Deep ensemble", "Majority label vote plus mean calibrated knownness", "Best accuracy/macro-F1; extra inference cost"],
    ], colWidths=[34*mm,79*mm,57*mm], repeatRows=1)
    models.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(BLUE)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("FONTSIZE",(0,0),(-1,-1),7.25),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story += [pipeline,
        Paragraph("4. End-to-End System Architecture", styles["H1X"]), Paragraph(
        "Figure 1 presents the complete learning and inference architecture. Encrypted packet or flow records are converted into privacy-preserving bidirectional flows and synchronized sequence, aggregate, and path-signature views. Three causal encoders provide complementary state-space, attention, and recurrent representations under an identical prototype and calibration protocol.", styles["BodyX"]),
        Image(str(chart_architecture()), width=166*mm, height=87*mm),
        Paragraph("Figure 1. NetAnomaly-OW architecture from encrypted flow ingestion to calibrated operational decisions.", styles["Cap"]), Paragraph(
        "The encoders fuse masked sequence summaries with aggregate and signature projections before mapping each flow to a normalized 64-dimensional embedding. Two prototypes per known application model different traffic modes. Split-conformal calibration converts class evidence into prediction sets and knownness p-values; the optional heterogeneous ensemble aggregates the three application votes and calibrated p-values. Drift monitors operate alongside inference and do not update model weights on test traffic.", styles["BodyX"]),
        KeepTogether([Paragraph("5. Multi-View Flow Representation", styles["H1X"]), Paragraph(
        "Packet payloads are never decrypted. The first packet positions are normalized into a causal tensor and accompanied by a mask so padded positions cannot influence pooling. Aggregate features provide global context, while second-order path signatures encode ordered cross-channel interactions. This combination follows the contextual-flow idea while remaining deployable from metadata.", styles["BodyX"])]), features,
        Paragraph("6. Neural Pipelines and Model Roles", styles["H1X"]), models, Paragraph(
        "All three encoders project packet observations into a shared latent dimension, apply masked mean and last-valid-state pooling, fuse aggregate and signature projections, and normalize the final embedding. Each application owns multiple learnable prototypes so one service can express several traffic modes. Similarity to the nearest class prototype produces logits and a hard-negative margin encourages separation from confusing applications.", styles["BodyX"]),
        Paragraph("7. Detailed Neural Architecture", styles["H1X"]), Paragraph(
        "The shared input is a packet tensor of shape [batch, 64, 3] containing normalized signed size, direction and inter-arrival time, paired with a [batch, 64] validity mask. Packet observations are projected to 64 dimensions and processed by three causal blocks. Masked mean pooling and the last valid hidden state are fused with independently projected aggregate features and six-dimensional path-signature features. The fused representation is normalized to a 64-dimensional embedding; each known application is represented by two learnable prototypes to capture multiple behavioral modes.", styles["BodyX"])]
    mechanics_data = [
        ["Component", "Implemented mechanism", "Computational / modeling implication"],
        ["DriftMamba", "Input-dependent candidate, write, read and step gates; softplus-positive decay; masked recurrent state scan", "Causal O(L) sequence processing and strongest unknown AUROC. It is Mamba-inspired pure PyTorch, not binary-equivalent to the official CUDA kernels."],
        ["Causal Transformer", "Learned positions, four-head masked attention, key-padding mask, three pre-norm layers and GELU feed-forward blocks", "Global packet interactions and highest individual accuracy, with O(L^2) attention cost."],
        ["xLSTM-inspired", "Stabilized exponential input/forget gates, running stabilizer and normalized scalar memory", "Causal O(L) recurrence with good early-packet behavior; inspired by xLSTM rather than a full official-package reproduction."],
        ["Prototype head", "Cosine similarity to two prototypes per class plus hard-negative separation", "Supports multimodal application behavior and produces interpretable nearest-prototype evidence."],
    ]
    mechanics_data = [[Paragraph(value, styles["CellHead"] if row_index == 0 else styles["Cell"])
                       for value in row] for row_index, row in enumerate(mechanics_data)]
    mechanics = Table(mechanics_data, colWidths=[31*mm,88*mm,51*mm], repeatRows=1)
    mechanics.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("FONTSIZE",(0,0),(-1,-1),7.15),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    decision_data = [
        ["Stage", "Configuration", "Rationale / emitted evidence"],
        ["Optimization", "AdamW; learning rate 1e-3; weight decay 1e-4; batch 128; gradient norm 1.0", "Stable mini-batch fitting; the checkpoint is selected by calibration loss rather than test performance."],
        ["Objective", "Cross-entropy + 0.25 x hard-negative prototype margin; margin 0.15", "Combines closed-set discrimination with explicit separation from the nearest incorrect application."],
        ["Conformal calibration", "1,000 disjoint known flows; alpha = 0.10", "Class p-values define the prediction set. A knownness p-value below 0.10 yields UNKNOWN."],
        ["Deep ensemble", "Majority application vote; arithmetic mean of three calibrated knownness p-values", "Combines complementary inductive biases; it is heterogeneous, not repeated seeds of one model."],
        ["Operational record", "Label, p-value, prediction set/size, nearest-prototype similarity, embedding norm and decision", "Preserves enough evidence for alert review, uncertainty analysis and drift monitoring."],
    ]
    decision_data = [[Paragraph(value, styles["CellHead"] if row_index == 0 else styles["Cell"])
                      for value in row] for row_index, row in enumerate(decision_data)]
    decision = Table(decision_data, colWidths=[32*mm,71*mm,67*mm], repeatRows=1)
    decision.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(TEAL)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("FONTSIZE",(0,0),(-1,-1),7.15),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story += [mechanics, Paragraph("8. Optimization and Open-World Decision Logic", styles["H1X"]), Paragraph(
        "Weights are fitted only on the training partition; calibration data determine stopping and the empirical conformal score distribution, while the test partitions remain untouched. Prefix inference at 8, 16, 32 and 64 packets measures how quickly a reliable decision becomes possible. Population Stability Index over input features and movement of the embedding centroid provide complementary chronological drift signals.", styles["BodyX"]), decision, Paragraph(
        "Reproducibility artifacts include the neural checkpoint, aggregate preprocessor, conformal calibrator, per-flow prediction CSV and metrics JSON. Together these separate learned parameters, calibration state and evaluation evidence, allowing the same operating threshold to be audited or changed without retraining the encoder.", styles["BodyX"]),
        Paragraph("9. Comparative Model Performance", styles["H1X"]), Paragraph(
        "Accuracy measures total known-flow correctness; balanced accuracy weights every application equally; macro-F1 emphasizes minority classes; unknown AUROC evaluates threshold-independent separation between known and held-out applications. No single metric is sufficient for an open-world classifier.", styles["BodyX"]),
        Image(str(chart_comparison()), width=170*mm, height=70*mm), Paragraph("Figure 2. Final four-pipeline comparison on the same held-out flows.", styles["Cap"])]
    rows = [["Pipeline","Accuracy","Balanced acc.","Macro-F1","Unknown AUROC","Best use"],
            ["Deep ensemble","0.823","0.7191","0.7264","0.6433","Overall classification"],
            ["Transformer","0.806","0.7046","0.6981","0.6090","Raw accuracy"],
            ["xLSTM-inspired","0.799","0.7087","0.6901","0.5643","Early classification"],
            ["DriftMamba-12","0.755","0.6253","0.6435","0.7011","Unknown separation"]]
    table = Table(rows, colWidths=[34*mm,22*mm,26*mm,23*mm,28*mm,37*mm], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("FONTSIZE",(0,0),(-1,-1),7.7),("ALIGN",(1,1),(4,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [table, Paragraph("The ensemble is heterogeneous: majority application voting across DriftMamba, Transformer and xLSTM, plus mean calibrated knownness. It is not three random seeds of one architecture. The ensemble maximizes classification quality, whereas DriftMamba provides the strongest neural unknown separation.", styles["BodyX"])]
    story += [Paragraph("10. Open-World Flagging and Operational Risk", styles["H1X"]), Paragraph(
        "A flag means the calibrated known-traffic p-value is below 0.10. It is a novelty/review decision, not proof of an attack. Known flagged flows represent false rejection or rare legitimate behavior; unknown flagged flows represent successful detection of a held-out service. The table therefore separates alert volume from alert utility.", styles["BodyX"])]
    flags = [["Pipeline","All flagged","Known flagged","Unknown flagged","Not flagged","Unknown recall","Known acceptance"],
             ["DriftMamba","402","110","292","1,598","29.2%","89.0%"],
             ["Transformer","278","88","190","1,722","19.0%","91.2%"],
             ["xLSTM-inspired","237","92","145","1,763","14.5%","90.8%"],
             ["Deep ensemble","167","51","116","1,833","11.6%","94.9%"]]
    ft = Table(flags, colWidths=[31*mm,20*mm,23*mm,24*mm,22*mm,25*mm,25*mm], repeatRows=1)
    ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(BLUE)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("FONTSIZE",(0,0),(-1,-1),7.4),("ALIGN",(1,1),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [ft, Spacer(1,1*mm), KeepTogether([
        Image(str(chart_flags()), width=170*mm, height=39*mm),
        Paragraph("Figure 3. Composition of model flags. More flags are not automatically better.", styles["Cap"]),
    ])]
    story += [Paragraph("11. Reproducibility and Repository Deliverables", styles["H1X"]), Paragraph(
        "The repository separates reusable source code, experiment configuration, validation, analysis notebooks, saved evidence and technical documentation. This structure supports transparent reruns, focused review and extension of each modelling pipeline.", styles["BodyX"])]
    deliverables = [["Deliverable","Repository location","Purpose"],
                    ["Python package","src/driftmamba","Data, models, calibration and inference"],
                    ["Automated verification","tests/ and .github/workflows/ci.yml","Regression tests and lint checks"],
                    ["Reproducible analysis","notebooks/ and results/","Data plots and benchmark comparisons"],
                    ["Experiment automation","scripts/ and configs/","Dataset export, training and report generation"],
                    ["Technical evidence","output/pdf/ and docs/","Architecture, results, limitations and handoff"]]
    rt = Table(deliverables, colWidths=[40*mm,60*mm,70*mm], repeatRows=1)
    rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("FONTSIZE",(0,0),(-1,-1),7.5),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [rt, Paragraph("12. Evidence Gaps and Threats to Validity", styles["H1X"]), Paragraph(
        "The present results are genuine but bounded: they use one dataset edition, one chronological transition, and one random seed. The following evidence is required before making strong generalization or deployment claims.", styles["BodyX"])]
    missing = [["Priority","Missing comparison parameter","Why it matters"],
               ["High","3-5 random seeds with confidence intervals","Current values are single-seed estimates"],
               ["High","Inference latency, throughput, memory and parameter count","Tests real-time feasibility"],
               ["High","Per-class precision/recall/F1 and support","Exposes minority-service failures"],
               ["High","Threshold curves: ROC, precision-recall and risk-coverage","Avoids one-threshold conclusions"],
               ["Medium","External validation, ablations and ensemble agreement","Tests generalization and novelty claims"]]
    mt = Table(missing, colWidths=[20*mm,75*mm,75*mm], repeatRows=1)
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(TEAL)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),("FONTSIZE",(0,0),(-1,-1),7.4),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
    story += [mt, Paragraph("13. Conclusions and Recommended Operating Modes", styles["H1X"]), Paragraph(
        "The heterogeneous ensemble is the preferred classification mode, reaching 0.823 accuracy and 0.7264 macro-F1. DriftMamba-12 is the unknown-sensitive option, with 0.7011 unknown AUROC and 292 held-out detections at 89.0% known acceptance. The Transformer prioritizes raw accuracy, while xLSTM is useful at early packet budgets. The large not-flagged count reflects conservative rejection and should change only after measuring known-flow false rejection. The defensible contribution is a complementary four-pipeline comparison under a leakage-free open-world protocol, not production attack attribution.", styles["BodyX"]), Paragraph(
        "NetAnomaly-OW | Results generated from saved experiment artifacts.", styles["Cap"])]
    doc.build(story)


if __name__ == "__main__":
    build()
