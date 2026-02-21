📘 CurricuForge AI
Generative AI Powered Curriculum Generation System (Colab Version)
🚀 Overview

CurricuForge AI is a Generative AI-based system that automatically creates structured academic curricula using Google Gemini.

The system generates:

Course learning outcomes

Weekly topic breakdown

Assignments

Final project structure

Standardized JSON output

It demonstrates practical LLM integration for real-world educational automation.

🎯 Problem Statement

Curriculum development in academic institutions is often:

Manual and time-consuming

Inconsistent across departments

Not standardized in structure

Difficult to scale

CurricuForge AI automates structured curriculum generation using large language models while enforcing strict JSON formatting for consistency and reproducibility.

🛠️ Tech Stack

Python

Google Generative AI SDK

Gemini API

vs code

⚙️ System Architecture

User Input
→ Prompt Engineering Layer
→ Gemini Model
→ Structured JSON Curriculum Output
→ Saved as .json file

This project focuses on clean AI integration rather than complex multi-service architecture.

📦 Features

Automated 5-days curriculum generation

Structured JSON output enforcement

Learning outcomes generation

Weekly breakdown with assignments

Final project suggestion

Clean, reproducible Colab implementation

📄 Example Output Structure
{
  "subject": "Maths",
  "duration": "5 days",
  "topics": [],
  "plan type": [
    {
      "week": 1,
      "topics": [],
      "assignment": ""
    }
  ],
  "final_project": ""
}
