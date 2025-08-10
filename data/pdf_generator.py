from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("science.pdf")
styles = getSampleStyleSheet()
story = []

content = [
    "The Earth revolves around the Sun once every 365.25 days, causing the seasons.",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius under standard atmospheric pressure.",
    "The speed of light in vacuum is approximately 299,792 kilometers per second.",
    "Gravity is a force that attracts two bodies toward each other. On Earth, it gives weight to objects.",
    "Photosynthesis is the process by which green plants and some organisms use sunlight to synthesize foods from carbon dioxide and water.",
    "The human body is made up of approximately 60% water.",
    "Electricity is the flow of electric charge, often carried by moving electrons in a conductor.",
    "Sound travels faster in water than in air due to higher density."
]

story.append(Paragraph("Science Facts", styles["Heading1"]))
story.append(Spacer(1, 12))
for fact in content:
    story.append(Paragraph(fact, styles["Normal"]))
    story.append(Spacer(1, 8))

doc.build(story)
