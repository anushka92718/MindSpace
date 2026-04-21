# 🧠 MindSpace v3 — Student Mental Health Platform

A premium, full-featured AI mental wellness web app for students.
Built with Flask · SQLite · scikit-learn · Chart.js · DM Fonts

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain ML model — model.pkl already included
python ml/train_model.py

# 3. Run the app
python app.py
```

Open → **http://localhost:5000**

---

## 🤖 ML Model (v2 — Upgraded)

| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| Decision Tree (tuned) | ~96% | ~95% |
| **Random Forest ✅** | **~98%** | **~98%** |
| Gradient Boosting | ~98% | ~98% |

**Selected: Random Forest (150 trees)**

Key improvements over v1:
- Class-aware data generation (fixes CLT collapse — v1 had only 2 Low/Critical samples)
- Balanced 500 samples per class (2000 total)
- Feature engineering: 4 category averages added (24 features total)
- Stratified 5-fold cross-validation
- Auto-selects best model by CV score

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 AI Assessment | 20-question ML assessment across Stress, Anxiety, Mood & Lifestyle |
| 📅 Daily Check-In | 5-metric daily logging (Mood, Stress, Sleep, Energy, Focus) |
| 📊 Dashboard | 4 tabs: Weekly (3 charts), Monthly (2 charts), Insights, Overview |
| 📓 Journal | Rich journaling with mood tags, writing prompts, word count |
| 🎯 Goals | Goal creation, quick templates, daily check-in streaks |
| 🧘 Meditation | 6 guided exercises with live countdown timer |
| 📅 Calendar | Mood history heatmap with month navigation |
| 💬 Community | Anonymous peer support posts with categories and likes |
| 🤖 Aria AI | Rule-based AI companion for stress, anxiety, mood & sleep |
| 📋 Report | Wellness report with charts + CSV export |
| 👤 Profile | Avatar, settings, dark mode, achievement badges (15 total) |
| 🔔 Notifications | Daily reminders + wellness tips |
| 🌟 Onboarding | 5-step animated welcome tour for new users |

---

## 📁 Structure

```
mindspace/
├── app.py                    ← 1000+ line Flask backend (33 routes)
├── model.pkl                 ← Random Forest, 98.25% accuracy
├── database.db               ← SQLite (auto-created on first run)
├── requirements.txt
├── README.md
├── templates/                ← 19 HTML templates
│   ├── base.html, _navbar.html
│   ├── login.html, assessment.html, result.html
│   ├── dashboard.html, calendar.html, report.html
│   ├── journal.html, journal_new.html, journal_view.html
│   ├── goals.html, meditation.html, community.html
│   ├── chat.html, profile.html, notifications.html
│   ├── onboarding.html, 404.html
├── static/
│   ├── css/style.css         ← 2100+ lines design system
│   └── js/script.js
└── ml/
    └── train_model.py        ← Upgraded ML pipeline
```

---

## 🗄️ Database (10 tables)

`users` · `assessments` · `daily_logs` · `journal_entries` · `goals`
`goal_checkins` · `community_posts` · `post_likes` · `chat_messages` · `meditation_logs`

---

## 🌐 Routes (33 total)

Auth, Assessment, Result, Daily Check-in, Dashboard, Journal (3),
Goals (5), Meditation, Calendar, Community (3), Chat (3),
Report, CSV Export, Profile, Notifications, Onboarding, APIs (2), Error handlers

---

## 📞 Crisis Resources (India)
- **iCall**: 9152987821
- **Vandrevala Foundation**: 1860-2662-345 (24/7)
- **NIMHANS**: 080-46110007
