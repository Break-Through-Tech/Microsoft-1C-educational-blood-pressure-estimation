---
title: CufflessAI
emoji: 🫀
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
short_description: Educational blood pressure estimates from a PPG pulse signal
---

# CufflessAI

An educational prototype that estimates systolic and diastolic blood pressure from five seconds of a PPG
pulse signal, with a range that shows how far off the estimate can be. Built by a student team in Break
Through Tech AI Studio, hosted by Microsoft.

**Not a medical device.** Don't use it to diagnose, monitor or treat any condition.

- Data: [UCI Cuff-Less Blood Pressure Estimation dataset](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)
- Signals you upload or capture with the camera are processed in memory and never stored. Camera video
  never leaves your browser.
- Without the trained result files, the app runs in demo mode with synthetic data.

This Space is deployed from the team's GitHub repository with `scripts/deploy_hf.py`; edit the code there,
not here.
