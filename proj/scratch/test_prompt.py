import os
import dotenv
dotenv.load_dotenv(".env")
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt = """
Textbook Page Context:
"Figure 3.1: Water Scarcity - Graph showing seasonal rainfall distribution and groundwater depletion levels across states in India."
Marker label: "Figure 3.1"

Task:
Analyze this figure based on page context and label.
Return JSON with keys:
{
  "figure_type": "graph",
  "description": "Bar chart illustrating seasonal rainfall distribution and groundwater depletion levels across Indian states.",
  "graph_type": "bar_chart",
  "x_axis_label": "States in India",
  "y_axis_label": "Groundwater level (m)",
  "trend_pattern": "Groundwater depletion increases in arid states"
}
"""

resp = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1
)
print("RESPONSE:\n", resp.choices[0].message.content)
