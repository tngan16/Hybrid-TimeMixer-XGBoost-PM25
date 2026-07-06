"""Create a PDF rendering that mirrors the editable Draw.io framework source."""
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "00_framework.pdf"

PAGE = (1224, 360)
BOXES = [
    (18, 126, 136, 86, "#DAE8FC",
     ["Five station CSVs", "MY1 | BIR | HP1", "CHB | KC1"]),
    (174, 126, 136, 86, "#FFF2CC",
     ["Audit and cleaning", "gap-aware windows"]),
    (330, 126, 136, 86, "#D5E8D4",
     ["TimeMixer", "five forecasts", "+ embedding"]),
    (486, 112, 158, 114, "#F8CECC",
     ["Robust residual XGBoost", "median de-bias", "pseudo-Huber loss",
      "lag/rolling context", "bounded correction"]),
    (664, 112, 150, 114, "#E1D5E7",
     ["Held-out correction gate", "alpha per horizon", "zero = base fallback"]),
    (834, 119, 136, 100, "#FFE6CC",
     ["Rolling calibration", "matured errors only", "30-day window"]),
    (990, 119, 136, 100, "#F5F5F5",
     ["Matched evaluation", "5 stations x 5 seeds", "30-epoch maximum"]),
]


def centered_lines(pdf, x, y, width, height, lines):
    line_height = 13
    top = y + height / 2 + line_height * (len(lines) - 1) / 2 - 4
    for index, line in enumerate(lines):
        pdf.drawCentredString(x + width / 2, top - index * line_height, line)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUT), pagesize=PAGE)
    pdf.setTitle("Calibration-gated TimeMixer-XGBoost framework")
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawCentredString(
        PAGE[0] / 2, 315,
        "Calibration-gated TimeMixer-XGBoost multi-station workflow",
    )
    pdf.setFont("Helvetica", 9)
    for x, y, width, height, color, lines in BOXES:
        pdf.setFillColor(HexColor(color))
        pdf.setStrokeColor(HexColor("#44546A"))
        pdf.roundRect(x, y, width, height, 8, fill=1, stroke=1)
        pdf.setFillColor(HexColor("#17202A"))
        centered_lines(pdf, x, y, width, height, lines)
    pdf.setStrokeColor(HexColor("#44546A"))
    pdf.setFillColor(HexColor("#44546A"))
    for left, right in zip(BOXES[:-1], BOXES[1:]):
        x1 = left[0] + left[2]
        x2 = right[0]
        y = 169
        pdf.line(x1 + 3, y, x2 - 8, y)
        pdf.line(x2 - 8, y, x2 - 15, y + 4)
        pdf.line(x2 - 8, y, x2 - 15, y - 4)
    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawCentredString(
        PAGE[0] / 2, 72,
        "Training, residual fit, gate selection, and locked testing are chronological.",
    )
    pdf.showPage()
    pdf.save()
    print(OUT)


if __name__ == "__main__":
    main()
