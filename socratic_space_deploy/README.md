---
title: Socratic Science Tutor
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 1h
short_description: Grade 10 Socratic Science Tutor powered by Qwen2.5-7B
pinned: false
---

# 🎓 Socratic Science Tutor (Grade 10)

Live demo for [Susu11/socratic_qwen8b](https://huggingface.co/Susu11/socratic_qwen8b), serving weights from the merged 16-bit standalone model [Susu11/socratic_qwen8b-merged](https://huggingface.co/Susu11/socratic_qwen8b-merged).

### 🌟 Key Pedagogical Capabilities
- **Socratic Inquiry**: Never dumps the final answer; guides the student step-by-step through inquiry.
- **Pedagogical Strategy**: Formulates teaching strategy and student misconception analysis inside `<plan>` tags.
- **Grade 10 Curriculum**: Physics (mechanics, electricity, light), Chemistry (reactions, bonding, periodic table), and Biology (genetics, ecology, human body).
- **ZeroGPU Acceleration**: Deployed on ZeroGPU with dynamic NVIDIA Blackwell GPU allocation.
