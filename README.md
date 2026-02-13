## Interactive Feature Presentation

This project includes an interactive, scroll-based infographic for presenting the Pump and Dump & Market Manipulation Detection system architecture.

How to Run the Presentation

To ensure all interactive charts (Chart.js and Plotly) render correctly, it is recommended to run the file through a local web server rather than opening it directly.
1. clone this repo and then

1. Using Python (Simplest)
Open your terminal and run the following command from the root directory:

python -m http.server 8000


Then, open your browser to:
http://localhost:8000/presentations/feature.html

2. Using VS Code
If you have the Live Server extension installed:

Right-click presentations/feature.html in the file explorer.

Select "Open with Live Server".

Presentation Controls

Scrolling: The presentation uses a smooth, continuous scroll with a progress bar at the top.

Keyboard Navigation: Use the Right/Down Arrow keys to move forward and Left/Up Arrow keys to move backward.

Sidebar Nav: Use the interactive dots on the right side to jump directly to specific technical sections:

01. Intent Source (Telegram Registry)

02. Market Reality (Binance Trade Data)

03. Feature Engineering (Logic & Normalization)

04. Feature Dictionary (Technical Variable Explanations)

05. Labeled Dataset (Final ML-Ready Output)

Technical Details Included

Microsecond Trade Logs: Visualizes raw execution data.

Standard Deviation Analysis: Explains the std_rush_order bot-detection logic.

Radar Signatures: Compares manipulation profiles against organic trading baselines.
