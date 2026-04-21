import os, pickle, json, sqlite3, hashlib, secrets
from datetime import date, timedelta, datetime
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, make_response)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import io, csv

app = Flask(__name__)
app.secret_key = os.urandom(24)

DB_PATH    = os.path.join(os.path.dirname(__file__), 'database.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# ── Model ──────────────────────────────────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

clf        = model_data['model']
dataset_X  = model_data['dataset_X']
dataset_y  = model_data['dataset_y']
weights    = model_data['weights']

FEATURE_NAMES = (
    [f"Stress {i}"    for i in range(1, 6)] +
    [f"Anxiety {i}"   for i in range(1, 6)] +
    [f"Mood {i}"      for i in range(1, 6)] +
    [f"Lifestyle {i}" for i in range(1, 6)]
)
RISK_LABELS = ['Low', 'Moderate', 'High', 'Critical']
RISK_COLORS = ['#10B981', '#F59E0B', '#F97316', '#EF4444']

MEDITATIONS = [
    {'id':1,'title':'Box Breathing','duration':'4 min','category':'Breathing',
     'icon':'🌬️','desc':'Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Activates the parasympathetic nervous system instantly.',
     'steps':['Sit comfortably with spine straight','Inhale slowly for 4 counts','Hold breath for 4 counts','Exhale gently for 4 counts','Hold empty for 4 counts','Repeat 4–6 cycles'],'color':'#7C6EFA'},
    {'id':2,'title':'5-4-3-2-1 Grounding','duration':'5 min','category':'Anxiety Relief',
     'icon':'🌿','desc':'Name 5 things you see, 4 you hear, 3 you can touch, 2 you smell, 1 you taste. Brings you back to the present.',
     'steps':['Find a comfortable position','Name 5 things you can see right now','Name 4 sounds you can hear','Touch 3 objects near you','Identify 2 scents in your environment','Notice 1 taste in your mouth'],'color':'#00BFA6'},
    {'id':3,'title':'Progressive Muscle Relaxation','duration':'8 min','category':'Stress Relief',
     'icon':'💪','desc':'Tense and release each muscle group from toes to head, releasing physical stress stored in the body.',
     'steps':['Lie down or sit comfortably','Tense feet muscles for 5 seconds, release','Move up to calves — tense and release','Continue: thighs, abdomen, hands, arms','Tense shoulders up to ears, release','Finish with face muscles — scrunch and release'],'color':'#FF6584'},
    {'id':4,'title':'4-7-8 Breath','duration':'3 min','category':'Breathing',
     'icon':'🌙','desc':'Dr. Weil\'s signature technique. Inhale for 4, hold for 7, exhale for 8. Reduces anxiety within minutes.',
     'steps':['Touch tongue tip to roof of mouth behind front teeth','Exhale completely through mouth','Inhale quietly through nose for 4 counts','Hold breath for 7 counts','Exhale forcefully through mouth for 8 counts','Repeat 3–4 cycles'],'color':'#7C6EFA'},
    {'id':5,'title':'Body Scan','duration':'10 min','category':'Mindfulness',
     'icon':'🧘','desc':'Systematically bring awareness to each part of your body without judgement. Builds mind-body connection.',
     'steps':['Lie flat on your back, close your eyes','Begin with the top of your head','Slowly move attention down: face, neck, shoulders','Continue through chest, abdomen, back','Move to hips, thighs, knees','Finish at feet — breathe and rest'],'color':'#F59E0B'},
    {'id':6,'title':'Loving Kindness','duration':'7 min','category':'Mood Lift',
     'icon':'💙','desc':'Send compassion to yourself first, then others. Clinically proven to reduce self-criticism and increase positive emotions.',
     'steps':['Sit comfortably and close eyes','Picture yourself clearly in your mind','Silently repeat: "May I be happy, may I be well"','Extend to someone you love','Extend to a neutral person','Finally extend to all living beings'],'color':'#00BFA6'},
]

GOALS_TEMPLATES = [
    {'title':'Sleep 7+ hours','category':'Lifestyle','icon':'😴','target_days':7},
    {'title':'No social media before bed','category':'Lifestyle','icon':'📵','target_days':14},
    {'title':'Exercise 3x per week','category':'Physical','icon':'🏃','target_days':21},
    {'title':'Journal daily','category':'Mental','icon':'📓','target_days':7},
    {'title':'Meditate daily','category':'Mental','icon':'🧘','target_days':7},
    {'title':'Drink 8 glasses of water','category':'Physical','icon':'💧','target_days':7},
    {'title':'Take a 10-min walk daily','category':'Physical','icon':'🌳','target_days':14},
    {'title':'Call a friend this week','category':'Social','icon':'📞','target_days':7},
]

# ── DB ─────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
        avatar_color TEXT DEFAULT "#7C6EFA",
        course TEXT DEFAULT "", university TEXT DEFAULT "",
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, score REAL, risk_level TEXT, responses TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS daily_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        log_date DATE NOT NULL DEFAULT CURRENT_DATE,
        mood INTEGER NOT NULL, stress INTEGER NOT NULL, sleep INTEGER NOT NULL,
        energy INTEGER DEFAULT 3, focus INTEGER DEFAULT 3,
        risk_score REAL, note TEXT DEFAULT "",
        UNIQUE(user_id, log_date),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS journal_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT "Untitled",
        content TEXT NOT NULL,
        mood_tag TEXT DEFAULT "neutral",
        tags TEXT DEFAULT "[]",
        is_private INTEGER DEFAULT 1,
        word_count INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL, category TEXT DEFAULT "General",
        icon TEXT DEFAULT "🎯",
        description TEXT DEFAULT "",
        target_days INTEGER DEFAULT 7,
        start_date DATE,
        end_date DATE,
        status TEXT DEFAULT "active",
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS goal_checkins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        checkin_date DATE DEFAULT CURRENT_DATE,
        completed INTEGER DEFAULT 1,
        note TEXT DEFAULT "",
        UNIQUE(goal_id, checkin_date),
        FOREIGN KEY (goal_id) REFERENCES goals(id)
    );
    CREATE TABLE IF NOT EXISTS community_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        category TEXT DEFAULT "General",
        likes INTEGER DEFAULT 0,
        is_anonymous INTEGER DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS post_likes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        UNIQUE(post_id, user_id)
    );
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS meditation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        meditation_id INTEGER NOT NULL,
        duration_seconds INTEGER DEFAULT 0,
        completed INTEGER DEFAULT 0,
        log_date DATE DEFAULT CURRENT_DATE,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ── Auth ───────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('assessment') if 'user_id' in session else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        action = request.form.get('action')
        email  = request.form.get('email','').strip()
        pwd    = request.form.get('password','')
        conn   = get_db()
        if action == 'signup':
            name       = request.form.get('name','').strip()
            course     = request.form.get('course','').strip()
            university = request.form.get('university','').strip()
            colors     = ['#7C6EFA','#00BFA6','#FF6584','#F59E0B','#3B82F6']
            color      = colors[len(name) % len(colors)]
            if not name or not email or not pwd:
                flash('All fields are required.','error')
            else:
                try:
                    conn.execute('INSERT INTO users (name,email,password,avatar_color,course,university) VALUES (?,?,?,?,?,?)',
                                 (name, email, generate_password_hash(pwd), color, course, university))
                    conn.commit()
                    flash('Account created! Please log in.','success')
                except sqlite3.IntegrityError:
                    flash('Email already registered.','error')
        else:
            user = conn.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone()
            if user and check_password_hash(user['password'], pwd):
                session.update({'user_id':user['id'],'user_name':user['name'],
                                'user_color':user['avatar_color'],'user_course':user['course']})
                conn.close(); return redirect(url_for('assessment'))
            flash('Invalid email or password.','error')
        conn.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('login'))

# ── Assessment ─────────────────────────────────────────────────────────────────
@app.route('/assessment')
@login_required
def assessment():
    conn = get_db()
    today_log = conn.execute(
        'SELECT * FROM daily_logs WHERE user_id=? AND log_date=date("now")',
        (session['user_id'],)).fetchone()
    conn.close()
    return render_template('assessment.html', questions=_get_questions(),
                           user_name=session['user_name'],
                           user_color=session.get('user_color','#7C6EFA'),
                           today_log=today_log)

@app.route('/result', methods=['POST'])
@login_required
def result():
    responses = [int(request.form.get(f'q{i}',3)) for i in range(1,21)]
    # Feature engineering: add 4 category averages (matches upgraded model)
    r = np.array(responses, dtype=float)
    engineered = np.array([r[0:5].mean(), r[5:10].mean(), r[10:15].mean(), r[15:20].mean()])
    X_in = np.concatenate([r, engineered]).reshape(1, -1)
    pred      = clf.predict(X_in)[0]
    probs     = clf.predict_proba(X_in)[0]
    score_raw = np.average(responses, weights=weights)
    score_pct = round((score_raw-1)/4*100, 1)
    feat_imp  = sorted(zip(FEATURE_NAMES, clf.feature_importances_.tolist()), key=lambda x:-x[1])
    cat_scores = {
        'Stress':    round(float(np.mean(responses[0:5])),  2),
        'Anxiety':   round(float(np.mean(responses[5:10])), 2),
        'Mood':      round(float(np.mean(responses[10:15])),2),
        'Lifestyle': round(float(np.mean(responses[15:20])),2),
    }
    unique, counts = np.unique(dataset_y, return_counts=True)
    risk_dist = {RISK_LABELS[int(u)]: int(c) for u,c in zip(unique,counts)}
    ds_scores = np.average(dataset_X, axis=1, weights=weights)
    hist_bins = np.histogram(ds_scores, bins=20)
    hist_data = {'counts': hist_bins[0].tolist(), 'edges': hist_bins[1].tolist()}
    conn = get_db()
    conn.execute('INSERT INTO assessments (user_id,score,risk_level,responses) VALUES (?,?,?,?)',
                 (session['user_id'], score_pct, RISK_LABELS[pred], json.dumps(responses)))
    conn.commit(); conn.close()
    return render_template('result.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        score=score_pct, risk_label=RISK_LABELS[pred], risk_color=RISK_COLORS[pred],
        risk_index=int(pred), probabilities=[round(p*100,1) for p in probs],
        cat_scores=cat_scores, feature_imp=feat_imp[:10],
        risk_dist=risk_dist, hist_data=hist_data,
        recommendations=get_recommendations(pred), responses=responses)

# ── Daily Check-in ─────────────────────────────────────────────────────────────
@app.route('/daily_checkin', methods=['POST'])
@login_required
def daily_checkin():
    mood   = max(1,min(5,int(request.form.get('mood',  3))))
    stress = max(1,min(5,int(request.form.get('stress',3))))
    sleep  = max(1,min(5,int(request.form.get('sleep', 3))))
    energy = max(1,min(5,int(request.form.get('energy',3))))
    focus  = max(1,min(5,int(request.form.get('focus', 3))))
    note   = request.form.get('note','')[:300]
    risk   = round((stress*0.35 + (6-mood)*0.30 + (6-sleep)*0.20 + (6-energy)*0.15)/5*100, 1)
    conn = get_db()
    try:
        conn.execute('''INSERT INTO daily_logs (user_id,log_date,mood,stress,sleep,energy,focus,risk_score,note)
                       VALUES (?,date("now"),?,?,?,?,?,?,?)''',
                     (session['user_id'],mood,stress,sleep,energy,focus,risk,note))
    except sqlite3.IntegrityError:
        conn.execute('''UPDATE daily_logs SET mood=?,stress=?,sleep=?,energy=?,focus=?,risk_score=?,note=?
                       WHERE user_id=? AND log_date=date("now")''',
                     (mood,stress,sleep,energy,focus,risk,note,session['user_id']))
    conn.commit(); conn.close()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status':'success','message':'Check-in saved!'})
    flash('Check-in saved!','success')
    return redirect(url_for('assessment'))

# ── Dashboard ──────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    uid  = session['user_id']
    conn = get_db()
    logs = conn.execute(
        '''SELECT log_date,mood,stress,sleep,energy,focus,risk_score,note
           FROM daily_logs WHERE user_id=? AND log_date>=date("now","-29 days")
           ORDER BY log_date ASC''', (uid,)).fetchall()
    last_assess = conn.execute(
        'SELECT score,risk_level,created_at FROM assessments WHERE user_id=? ORDER BY id DESC LIMIT 1',
        (uid,)).fetchone()
    active_goals = conn.execute(
        '''SELECT g.*, (SELECT COUNT(*) FROM goal_checkins gc WHERE gc.goal_id=g.id) as checkin_count
           FROM goals g WHERE g.user_id=? AND g.status="active" ORDER BY g.created_at DESC LIMIT 4''',
        (uid,)).fetchall()
    med_count = conn.execute(
        'SELECT COUNT(*) as c FROM meditation_logs WHERE user_id=? AND completed=1', (uid,)).fetchone()['c']
    journal_count = conn.execute(
        'SELECT COUNT(*) as c FROM journal_entries WHERE user_id=?',(uid,)).fetchone()['c']
    conn.close()

    if not logs:
        return render_template('dashboard.html', user_name=session['user_name'],
                               user_color=session.get('user_color','#7C6EFA'),
                               has_data=False, last_assessment=last_assess,
                               active_goals=active_goals, med_count=med_count,
                               journal_count=journal_count)

    rows = [dict(r) for r in logs]
    today = date.today()
    date_map = {r['log_date']: r for r in rows}

    def build_series(field, days):
        labels, vals = [], []
        for i in range(days-1, -1, -1):
            d = (today - timedelta(days=i))
            labels.append(d.strftime('%a' if days<=7 else '%d %b'))
            e = date_map.get(d.isoformat())
            vals.append(e[field] if e else None)
        return labels, vals

    wl, wm  = build_series('mood',   7); _, ws = build_series('stress', 7)
    _, wsl  = build_series('sleep',  7); _, we = build_series('energy', 7); _, wf = build_series('focus',7)
    ml, mm  = build_series('mood',  30); _, ms = build_series('stress', 30)
    _, msl  = build_series('sleep', 30)

    valid = rows
    avgs  = {k: round(sum(r[k] for r in valid)/len(valid),1)
             for k in ('mood','stress','sleep','energy','focus')}

    streak = 0
    for i in range(60):
        if (today-timedelta(days=i)).isoformat() in date_map: streak+=1
        else: break

    return render_template('dashboard.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        has_data=True, last_assessment=last_assess,
        avgs=avgs, streak=streak, total_entries=len(rows),
        best_day=max(rows,key=lambda r:r['mood']),
        worst_day=max(rows,key=lambda r:r['stress']),
        insights=_generate_insights(rows,today),
        prediction=_predict_next_week(rows,today),
        week_labels=wl, week_mood=wm, week_stress=ws,
        week_sleep=wsl, week_energy=we, week_focus=wf,
        month_labels=ml, month_mood=mm, month_stress=ms, month_sleep=msl,
        active_goals=[dict(g) for g in active_goals],
        med_count=med_count, journal_count=journal_count)

# ── Journal ────────────────────────────────────────────────────────────────────
@app.route('/journal')
@login_required
def journal():
    conn = get_db()
    entries = conn.execute(
        'SELECT * FROM journal_entries WHERE user_id=? ORDER BY created_at DESC',
        (session['user_id'],)).fetchall()
    conn.close()
    mood_counts = {}
    for e in entries:
        mood_counts[e['mood_tag']] = mood_counts.get(e['mood_tag'],0)+1
    return render_template('journal.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        entries=entries, mood_counts=mood_counts,
        total_words=sum(e['word_count'] for e in entries))

@app.route('/journal/new', methods=['GET','POST'])
@login_required
def journal_new():
    if request.method == 'POST':
        title   = request.form.get('title','Untitled')[:120]
        content = request.form.get('content','').strip()
        mood_tag= request.form.get('mood_tag','neutral')
        tags_raw= request.form.get('tags','')
        tags    = json.dumps([t.strip() for t in tags_raw.split(',') if t.strip()][:5])
        wc      = len(content.split())
        conn = get_db()
        conn.execute('INSERT INTO journal_entries (user_id,title,content,mood_tag,tags,word_count) VALUES (?,?,?,?,?,?)',
                     (session['user_id'],title,content,mood_tag,tags,wc))
        conn.commit(); conn.close()
        flash('Journal entry saved!','success')
        return redirect(url_for('journal'))
    return render_template('journal_new.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'))

@app.route('/journal/<int:entry_id>')
@login_required
def journal_view(entry_id):
    conn = get_db()
    entry = conn.execute(
        'SELECT * FROM journal_entries WHERE id=? AND user_id=?',
        (entry_id, session['user_id'])).fetchone()
    conn.close()
    if not entry: return redirect(url_for('journal'))
    return render_template('journal_view.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        entry=dict(entry), tags=json.loads(entry['tags'] or '[]'))

@app.route('/journal/<int:entry_id>/delete', methods=['POST'])
@login_required
def journal_delete(entry_id):
    conn = get_db()
    conn.execute('DELETE FROM journal_entries WHERE id=? AND user_id=?',
                 (entry_id, session['user_id']))
    conn.commit(); conn.close()
    return redirect(url_for('journal'))

# ── Goals ──────────────────────────────────────────────────────────────────────
@app.route('/goals')
@login_required
def goals():
    uid  = session['user_id']
    conn = get_db()
    goals_list = conn.execute(
        '''SELECT g.*,
           (SELECT COUNT(*) FROM goal_checkins gc WHERE gc.goal_id=g.id AND gc.completed=1) as done_count
           FROM goals g WHERE g.user_id=? ORDER BY g.status ASC, g.created_at DESC''',
        (uid,)).fetchall()
    today_checkins = set(row['goal_id'] for row in conn.execute(
        'SELECT goal_id FROM goal_checkins WHERE user_id=? AND checkin_date=date("now")', (uid,)).fetchall())
    conn.close()
    return render_template('goals.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        goals=goals_list, templates=GOALS_TEMPLATES, today_checkins=today_checkins)

@app.route('/goals/add', methods=['POST'])
@login_required
def goal_add():
    title  = request.form.get('title','').strip()[:100]
    cat    = request.form.get('category','General')
    icon   = request.form.get('icon','🎯')
    desc   = request.form.get('description','').strip()[:300]
    days   = max(1,min(90,int(request.form.get('target_days',7))))
    end_d  = (date.today()+timedelta(days=days)).isoformat()
    if title:
        conn = get_db()
        conn.execute('INSERT INTO goals (user_id,title,category,icon,description,target_days,end_date) VALUES (?,?,?,?,?,?,?)',
                     (session['user_id'],title,cat,icon,desc,days,end_d))
        conn.commit(); conn.close()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status':'success'})
    return redirect(url_for('goals'))

@app.route('/goals/<int:goal_id>/checkin', methods=['POST'])
@login_required
def goal_checkin(goal_id):
    note = request.form.get('note','')[:200]
    conn = get_db()
    try:
        conn.execute('INSERT INTO goal_checkins (goal_id,user_id,note) VALUES (?,?,?)',
                     (goal_id, session['user_id'], note))
        conn.commit()
        msg = 'Checked in!'
    except sqlite3.IntegrityError:
        msg = 'Already checked in today'
    conn.close()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status':'success','message':msg})
    return redirect(url_for('goals'))

@app.route('/goals/<int:goal_id>/complete', methods=['POST'])
@login_required
def goal_complete(goal_id):
    conn = get_db()
    conn.execute("UPDATE goals SET status='completed' WHERE id=? AND user_id=?",
                 (goal_id, session['user_id']))
    conn.commit(); conn.close()
    return redirect(url_for('goals'))

@app.route('/goals/<int:goal_id>/delete', methods=['POST'])
@login_required
def goal_delete(goal_id):
    conn = get_db()
    conn.execute('DELETE FROM goal_checkins WHERE goal_id=?', (goal_id,))
    conn.execute('DELETE FROM goals WHERE id=? AND user_id=?', (goal_id,session['user_id']))
    conn.commit(); conn.close()
    return redirect(url_for('goals'))

# ── Meditation ─────────────────────────────────────────────────────────────────
@app.route('/meditation')
@login_required
def meditation():
    uid  = session['user_id']
    conn = get_db()
    completed_ids = set(row['meditation_id'] for row in conn.execute(
        'SELECT DISTINCT meditation_id FROM meditation_logs WHERE user_id=? AND completed=1',(uid,)).fetchall())
    total_sessions = conn.execute(
        'SELECT COUNT(*) as c FROM meditation_logs WHERE user_id=? AND completed=1',(uid,)).fetchone()['c']
    total_mins = conn.execute(
        'SELECT COALESCE(SUM(duration_seconds),0) as s FROM meditation_logs WHERE user_id=? AND completed=1',(uid,)).fetchone()['s']
    conn.close()
    return render_template('meditation.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        meditations=MEDITATIONS, completed_ids=completed_ids,
        total_sessions=total_sessions, total_mins=total_mins//60)

@app.route('/meditation/log', methods=['POST'])
@login_required
def meditation_log():
    med_id   = int(request.form.get('meditation_id',1))
    duration = int(request.form.get('duration_seconds',0))
    completed= int(request.form.get('completed',0))
    conn = get_db()
    conn.execute('INSERT INTO meditation_logs (user_id,meditation_id,duration_seconds,completed) VALUES (?,?,?,?)',
                 (session['user_id'],med_id,duration,completed))
    conn.commit(); conn.close()
    return jsonify({'status':'success'})

# ── Community ──────────────────────────────────────────────────────────────────
@app.route('/community')
@login_required
def community():
    uid  = session['user_id']
    conn = get_db()
    cat  = request.args.get('cat','All')
    q    = 'SELECT cp.*, u.name FROM community_posts cp JOIN users u ON cp.user_id=u.id'
    if cat != 'All':
        posts = conn.execute(q+' WHERE cp.category=? ORDER BY cp.created_at DESC LIMIT 50',(cat,)).fetchall()
    else:
        posts = conn.execute(q+' ORDER BY cp.created_at DESC LIMIT 50').fetchall()
    liked_ids = set(row['post_id'] for row in conn.execute(
        'SELECT post_id FROM post_likes WHERE user_id=?',(uid,)).fetchall())
    cats = conn.execute('SELECT DISTINCT category FROM community_posts ORDER BY category').fetchall()
    conn.close()
    return render_template('community.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        posts=posts, liked_ids=liked_ids,
        categories=['All']+[c['category'] for c in cats],
        current_cat=cat)

@app.route('/community/post', methods=['POST'])
@login_required
def community_post():
    content = request.form.get('content','').strip()[:500]
    cat     = request.form.get('category','General')
    anon    = 1 if request.form.get('anonymous') else 0
    if content:
        conn = get_db()
        conn.execute('INSERT INTO community_posts (user_id,content,category,is_anonymous) VALUES (?,?,?,?)',
                     (session['user_id'],content,cat,anon))
        conn.commit(); conn.close()
    return redirect(url_for('community'))

@app.route('/community/like/<int:post_id>', methods=['POST'])
@login_required
def community_like(post_id):
    uid  = session['user_id']
    conn = get_db()
    try:
        conn.execute('INSERT INTO post_likes (post_id,user_id) VALUES (?,?)',(post_id,uid))
        conn.execute('UPDATE community_posts SET likes=likes+1 WHERE id=?',(post_id,))
        action = 'liked'
    except sqlite3.IntegrityError:
        conn.execute('DELETE FROM post_likes WHERE post_id=? AND user_id=?',(post_id,uid))
        conn.execute('UPDATE community_posts SET likes=MAX(0,likes-1) WHERE id=?',(post_id,))
        action = 'unliked'
    conn.commit()
    likes = conn.execute('SELECT likes FROM community_posts WHERE id=?',(post_id,)).fetchone()['likes']
    conn.close()
    return jsonify({'status':action,'likes':likes})

# ── AI Chat ────────────────────────────────────────────────────────────────────
CHAT_SYSTEM = """You are Aria, a warm, empathetic AI mental wellness companion for students. 
You provide supportive conversations about stress, anxiety, study pressure, and general wellbeing.
Keep responses concise (2-4 sentences), compassionate, and practical. 
Always remind users you're an AI and encourage professional help for serious issues.
Use a friendly, peer-like tone — not clinical. Occasionally use emojis.
Never diagnose. Focus on coping strategies, validation, and encouragement."""

ARIA_RESPONSES = {
    'stress': [
        "Exam stress is so real 😔 — your brain is in overdrive right now. Try the 5-4-3-2-1 grounding technique: name 5 things you can see, 4 you hear, 3 you can touch. It breaks the anxiety loop fast. You've handled tough stuff before — this too shall pass 💙",
        "Academic pressure can feel crushing, I hear you. One thing that helps: break your workload into the smallest possible chunks. Just focus on the next 25 minutes, nothing else. Progress beats perfection every time 🌟",
        "Feeling overwhelmed is your mind saying 'I need support.' That's valid! Have you tried a quick walk or even just stepping outside for 5 minutes? Fresh air genuinely shifts brain chemistry. Also — is there someone at your university you can talk to? 🌿"
    ],
    'anxiety': [
        "Anxiety loves to convince us the worst will happen — but it's usually lying 😮‍💨 Try box breathing: inhale 4 counts, hold 4, exhale 4, hold 4. Repeat 4 times. Your nervous system will actually calm down within minutes 🌬️",
        "That anxious spiral feeling is exhausting 💙 Your body thinks it's in danger when really it's just... Tuesday. Try placing one hand on your chest and breathing slowly — feeling your heartbeat can ground you back in the present moment.",
        "Anxiety before presentations or exams is extremely common — nearly 70% of students report it! What helps most: preparation + self-compassion. You're not failing, you're feeling. Big difference 🌟"
    ],
    'mood': [
        "Low mood days are part of being human, especially during demanding study periods 💙 Tiny acts matter — a playlist you love, texting a friend, or even just opening a window. What's one small thing that usually lifts you, even slightly?",
        "I'm glad you're talking about how you feel 🌿 When mood dips, our inner critic gets louder. Try this: write down 3 things — however tiny — that went okay today. It rewires the brain's negativity bias over time.",
        "Persistent low mood is worth paying attention to 💙 Have you been able to get outside or move your body a bit? Even a 10-minute walk releases endorphins. And please — if this feeling lasts more than 2 weeks, do reach out to a counsellor. You deserve support 🌟"
    ],
    'sleep': [
        "Sleep deprivation hits students hard 😴 Your brain literally can't consolidate memories or regulate emotions without it. Try: same wake time every day (yes, weekends too), no screens 30 min before bed, and keep your room cool. Which of these feels most doable?",
        "Pulling all-nighters actually backfires — you retain less and perform worse 😮‍💨 A 90-minute study + 20-minute break cycle is more effective. Your brain needs rest to process what you're learning 🌙",
        "Racing thoughts at bedtime? Try a 'brain dump' — write everything in your head onto paper before sleep. It signals to your brain 'this is handled, we can rest now.' Works surprisingly well 📓"
    ],
    'default': [
        "Thank you for sharing that with me 💙 I'm here to listen. Can you tell me a bit more about what's been going on? I want to make sure I understand what you're experiencing.",
        "That sounds really tough, and I want you to know your feelings are completely valid 🌿 Would it help to talk through what's been happening, or would you prefer some practical coping strategies?",
        "I hear you, and I'm glad you reached out 🌟 University life can be genuinely hard — academically, socially, everything at once. What feels most pressing for you right now?"
    ]
}

@app.route('/chat')
@login_required
def chat():
    uid  = session['user_id']
    conn = get_db()
    history = conn.execute(
        'SELECT role,content,created_at FROM chat_messages WHERE user_id=? ORDER BY created_at DESC LIMIT 40',
        (uid,)).fetchall()
    conn.close()
    return render_template('chat.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        history=list(reversed(history)))

@app.route('/chat/send', methods=['POST'])
@login_required
def chat_send():
    uid     = session['user_id']
    message = request.json.get('message','').strip()[:500]
    if not message:
        return jsonify({'error':'empty'}), 400

    # Save user message
    conn = get_db()
    conn.execute('INSERT INTO chat_messages (user_id,role,content) VALUES (?,?,?)',
                 (uid,'user',message))

    # Rule-based response selection
    msg_lower = message.lower()
    if any(w in msg_lower for w in ['stress','stressed','overwhelm','pressure','deadline','exam','test']):
        import random; response = random.choice(ARIA_RESPONSES['stress'])
    elif any(w in msg_lower for w in ['anxious','anxiety','panic','worry','nervous','fear']):
        import random; response = random.choice(ARIA_RESPONSES['anxiety'])
    elif any(w in msg_lower for w in ['sad','depressed','down','low','unmotivated','hopeless','empty']):
        import random; response = random.choice(ARIA_RESPONSES['mood'])
    elif any(w in msg_lower for w in ['sleep','tired','exhausted','insomnia','awake','fatigue']):
        import random; response = random.choice(ARIA_RESPONSES['sleep'])
    else:
        import random; response = random.choice(ARIA_RESPONSES['default'])

    conn.execute('INSERT INTO chat_messages (user_id,role,content) VALUES (?,?,?)',
                 (uid,'assistant',response))
    conn.commit(); conn.close()
    return jsonify({'response': response})

@app.route('/chat/clear', methods=['POST'])
@login_required
def chat_clear():
    conn = get_db()
    conn.execute('DELETE FROM chat_messages WHERE user_id=?',(session['user_id'],))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

# ── Calendar ───────────────────────────────────────────────────────────────────
@app.route('/calendar')
@login_required
def calendar_view():
    uid  = session['user_id']
    conn = get_db()
    logs = conn.execute(
        'SELECT log_date,mood,stress,sleep FROM daily_logs WHERE user_id=? ORDER BY log_date DESC LIMIT 90',
        (uid,)).fetchall()
    conn.close()
    cal_data = {r['log_date']: {'mood':r['mood'],'stress':r['stress'],'sleep':r['sleep']} for r in logs}
    return render_template('calendar.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        cal_data=cal_data, cal_json=json.dumps(cal_data))

# ── Reports & Export ───────────────────────────────────────────────────────────
@app.route('/report')
@login_required
def report():
    uid  = session['user_id']
    conn = get_db()
    logs = conn.execute(
        'SELECT * FROM daily_logs WHERE user_id=? ORDER BY log_date DESC',(uid,)).fetchall()
    assessments = conn.execute(
        'SELECT score,risk_level,created_at FROM assessments WHERE user_id=? ORDER BY created_at DESC',(uid,)).fetchall()
    user = conn.execute('SELECT * FROM users WHERE id=?',(uid,)).fetchone()
    j_count = conn.execute('SELECT COUNT(*) as c FROM journal_entries WHERE user_id=?',(uid,)).fetchone()['c']
    g_count = conn.execute('SELECT COUNT(*) as c FROM goals WHERE user_id=?',(uid,)).fetchone()['c']
    m_count = conn.execute('SELECT COUNT(*) as c FROM meditation_logs WHERE user_id=? AND completed=1',(uid,)).fetchone()['c']
    conn.close()

    avg_mood   = round(sum(r['mood']   for r in logs)/len(logs),1) if logs else 0
    avg_stress = round(sum(r['stress'] for r in logs)/len(logs),1) if logs else 0
    avg_sleep  = round(sum(r['sleep']  for r in logs)/len(logs),1) if logs else 0

    return render_template('report.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        user=user, logs=logs, assessments=assessments,
        avg_mood=avg_mood, avg_stress=avg_stress, avg_sleep=avg_sleep,
        j_count=j_count, g_count=g_count, m_count=m_count,
        report_date=date.today().strftime('%B %d, %Y'))

@app.route('/export/csv')
@login_required
def export_csv():
    uid  = session['user_id']
    conn = get_db()
    logs = conn.execute('SELECT * FROM daily_logs WHERE user_id=? ORDER BY log_date DESC',(uid,)).fetchall()
    conn.close()
    si  = io.StringIO()
    cw  = csv.writer(si)
    cw.writerow(['Date','Mood','Stress','Sleep','Energy','Focus','Risk Score','Note'])
    for r in logs:
        cw.writerow([r['log_date'],r['mood'],r['stress'],r['sleep'],
                     r['energy'] or '',r['focus'] or '',r['risk_score'] or '',r['note'] or ''])
    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = f'attachment; filename=mindspace_logs_{date.today()}.csv'
    output.headers['Content-type'] = 'text/csv'
    return output

# ── API ────────────────────────────────────────────────────────────────────────
@app.route('/weekly_data')
@login_required
def weekly_data():
    conn = get_db()
    rows = conn.execute(
        'SELECT log_date,mood,stress,sleep,energy,focus FROM daily_logs WHERE user_id=? AND log_date>=date("now","-6 days") ORDER BY log_date',
        (session['user_id'],)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/monthly_data')
@login_required
def monthly_data():
    conn = get_db()
    rows = conn.execute(
        'SELECT log_date,mood,stress,sleep FROM daily_logs WHERE user_id=? AND log_date>=date("now","-29 days") ORDER BY log_date',
        (session['user_id'],)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ── Helpers ────────────────────────────────────────────────────────────────────
def _generate_insights(rows, today):
    insights = []
    if len(rows) < 2:
        return [{'icon':'📊','text':'Keep logging daily to unlock trend insights.','type':'info'}]
    this  = [r for r in rows if r['log_date']>=(today-timedelta(days=6)).isoformat()]
    prev  = [r for r in rows if r['log_date']>=(today-timedelta(days=13)).isoformat()
                             and r['log_date']< (today-timedelta(days=6)).isoformat()]
    if this:
        as_ = lambda k,lst: sum(r[k] for r in lst)/len(lst)
        sn,mn,sln = as_('stress',this),as_('mood',this),as_('sleep',this)
        if prev:
            sp,mp = as_('stress',prev),as_('mood',prev)
            sd = (sn-sp)/sp*100; md = (mn-mp)/mp*100
            if sd>10:  insights.append({'icon':'⚠️','text':f'Stress rose {abs(sd):.0f}% vs last week. Consider scheduling a break.','type':'warning'})
            elif sd<-10: insights.append({'icon':'✅','text':f'Stress dropped {abs(sd):.0f}% — great progress this week!','type':'good'})
            else: insights.append({'icon':'📊','text':'Stress has been stable this week.','type':'info'})
            if md>8:   insights.append({'icon':'😊','text':f'Mood improved {abs(md):.0f}% from last week — keep it up!','type':'good'})
            elif md<-8: insights.append({'icon':'💙','text':f'Mood dipped {abs(md):.0f}% — consider reaching out to someone you trust.','type':'warning'})
        if sln<3 and sn>3: insights.append({'icon':'😴','text':'Poor sleep appears correlated with higher stress. Prioritise rest.','type':'warning'})
        elif sln>=4: insights.append({'icon':'🌟','text':'Excellent sleep quality — this is strongly supporting your mental health.','type':'good'})
        if len(this)>=5:
            recent_moods = [r['mood'] for r in this[-3:]]
            if all(m>=4 for m in recent_moods): insights.append({'icon':'🔥','text':'Your mood has been consistently high — you\'re on a great streak!','type':'good'})
    if not insights: insights.append({'icon':'📈','text':'Log more days to unlock personalised insights.','type':'info'})
    return insights

def _predict_next_week(rows, today):
    recent = [r for r in rows if r['log_date']>=(today-timedelta(days=6)).isoformat()]
    if len(recent)<3: return {'text':'Log at least 3 days to unlock your forecast.','type':'info'}
    ss = [r['stress'] for r in recent]; ms = [r['mood'] for r in recent]
    mid = len(ss)//2
    st = sum(ss[mid:])/len(ss[mid:]) - sum(ss[:mid])/max(len(ss[:mid]),1)
    mt = sum(ms[mid:])/len(ms[mid:]) - sum(ms[:mid])/max(len(ms[:mid]),1)
    if st>0.5: return {'text':'Stress is rising — plan lighter commitments next week and protect your sleep.','type':'warning'}
    elif st<-0.5 and mt>0: return {'text':'Positive trajectory! Mental state is improving — maintain your current habits.','type':'good'}
    elif mt<-0.5: return {'text':'Mood has been declining. Consider talking to a friend or counsellor this week.','type':'warning'}
    return {'text':'Your mental state appears stable. Consistency is key — keep logging!','type':'info'}

def _get_questions():
    return {
        'Stress':    ["I feel overwhelmed by my responsibilities.",
                      "I struggle to manage my time effectively.",
                      "I feel pressure from academics/work.",
                      "I have difficulty relaxing after a stressful day.",
                      "Minor setbacks feel like major crises to me."],
        'Anxiety':   ["I experience excessive worry about the future.",
                      "I have trouble concentrating due to anxious thoughts.",
                      "My heart races in social or academic situations.",
                      "I avoid situations that cause me anxiety.",
                      "I feel a constant sense of nervousness or unease."],
        'Mood':      ["I feel hopeless or empty inside.",
                      "I've lost interest in activities I used to enjoy.",
                      "I feel irritable or short-tempered.",
                      "I experience sudden mood swings.",
                      "I struggle to find motivation to start tasks."],
        'Lifestyle': ["My sleep schedule is irregular or disrupted.",
                      "I skip meals or eat unhealthily when stressed.",
                      "I rarely engage in physical exercise.",
                      "I feel socially isolated or disconnected.",
                      "I rely on screens/substances to cope with stress."]
    }

def get_recommendations(risk_level):
    base = [
        {'icon':'🧘','title':'Mindful Breathing','desc':'Practice 4-7-8 breathing for 5 minutes daily to activate your parasympathetic nervous system.'},
        {'icon':'📓','title':'Reflective Journaling','desc':'Write 3 things you are grateful for each morning to rewire positive neural pathways.'},
        {'icon':'🏃','title':'Physical Movement','desc':'Even 20 minutes of brisk walking releases endorphins and reduces cortisol significantly.'},
        {'icon':'😴','title':'Sleep Hygiene','desc':'Maintain a consistent sleep schedule. Aim for 7–9 hours with no screens 1 hour before bed.'},
    ]
    if risk_level>=1: base += [
        {'icon':'🗣','title':'Talk to Someone','desc':'Share your feelings with a trusted friend, mentor, or peer support group.'},
        {'icon':'🎨','title':'Creative Outlets','desc':'Channel emotions into art, music, or writing — proven to reduce stress hormones.'},
    ]
    if risk_level>=2: base += [
        {'icon':'🧑‍⚕️','title':'Counseling Session','desc':'Book a session with your institution\'s counselor. CBT shows strong results.'},
        {'icon':'📱','title':'Mental Health Apps','desc':'Use apps like Headspace or Calm for guided therapeutic exercises on-demand.'},
    ]
    if risk_level>=3: base += [
        {'icon':'🚨','title':'Seek Professional Help Now','desc':'Please contact a licensed mental health professional or crisis helpline immediately.'},
        {'icon':'📞','title':'Crisis Resources','desc':'iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345 | 24/7'},
    ]
    return base

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# ══════════════════════════════════════════════════════════════════════════════
#  v3.1 ADDITIONS — Profile, Achievements, Onboarding, Notifications, 404
# ══════════════════════════════════════════════════════════════════════════════

BADGES = [
    {'id':'first_checkin',  'icon':'🌱','title':'First Step',        'desc':'Completed your first daily check-in',          'color':'#10B981'},
    {'id':'streak_3',       'icon':'🔥','title':'3-Day Streak',      'desc':'Logged 3 days in a row',                       'color':'#F59E0B'},
    {'id':'streak_7',       'icon':'⚡','title':'Week Warrior',       'desc':'Logged 7 days in a row',                       'color':'#7C6EFA'},
    {'id':'streak_30',      'icon':'🏆','title':'Monthly Master',    'desc':'Logged 30 days in a row',                      'color':'#EF4444'},
    {'id':'journal_1',      'icon':'📓','title':'First Entry',       'desc':'Wrote your first journal entry',               'color':'#00BFA6'},
    {'id':'journal_10',     'icon':'✍️', 'title':'Storyteller',       'desc':'Wrote 10 journal entries',                     'color':'#7C6EFA'},
    {'id':'journal_1000',   'icon':'📚','title':'Wordsmith',         'desc':'Written over 1,000 words in your journal',     'color':'#F59E0B'},
    {'id':'goals_first',    'icon':'🎯','title':'Goal Setter',       'desc':'Created your first goal',                     'color':'#F97316'},
    {'id':'goals_complete', 'icon':'✅','title':'Achiever',          'desc':'Completed your first goal',                   'color':'#10B981'},
    {'id':'meditation_1',   'icon':'🧘','title':'Inner Peace',       'desc':'Completed your first meditation session',      'color':'#00BFA6'},
    {'id':'meditation_10',  'icon':'🌟','title':'Zen Student',       'desc':'Completed 10 meditation sessions',             'color':'#7C6EFA'},
    {'id':'community_1',    'icon':'💙','title':'Community Member',  'desc':'Made your first community post',               'color':'#3B82F6'},
    {'id':'assessment_1',   'icon':'🧠','title':'Self-Aware',        'desc':'Completed your first mental health assessment','color':'#FF6584'},
    {'id':'low_stress',     'icon':'😌','title':'Calm & Collected',  'desc':'Maintained low stress for 5+ days this week', 'color':'#10B981'},
    {'id':'mood_boost',     'icon':'📈','title':'Rising Star',       'desc':'Improved mood score 3 days in a row',         'color':'#F59E0B'},
]

def compute_badges(uid, conn):
    earned = set()
    logs = conn.execute('SELECT * FROM daily_logs WHERE user_id=? ORDER BY log_date',(uid,)).fetchall()
    journals = conn.execute('SELECT * FROM journal_entries WHERE user_id=?',(uid,)).fetchall()
    goals = conn.execute('SELECT * FROM goals WHERE user_id=?',(uid,)).fetchall()
    goals_done = conn.execute("SELECT COUNT(*) as c FROM goals WHERE user_id=? AND status='completed'",(uid,)).fetchone()['c']
    med = conn.execute('SELECT COUNT(*) as c FROM meditation_logs WHERE user_id=? AND completed=1',(uid,)).fetchone()['c']
    posts = conn.execute('SELECT COUNT(*) as c FROM community_posts WHERE user_id=?',(uid,)).fetchone()['c']
    assess = conn.execute('SELECT COUNT(*) as c FROM assessments WHERE user_id=?',(uid,)).fetchone()['c']

    if logs:        earned.add('first_checkin')
    if assess:      earned.add('assessment_1')
    if journals:    earned.add('journal_1')
    if len(journals) >= 10: earned.add('journal_10')
    if sum(j['word_count'] for j in journals) >= 1000: earned.add('journal_1000')
    if goals:       earned.add('goals_first')
    if goals_done:  earned.add('goals_complete')
    if med >= 1:    earned.add('meditation_1')
    if med >= 10:   earned.add('meditation_10')
    if posts:       earned.add('community_1')

    # Streak badges
    from datetime import date, timedelta
    today = date.today()
    date_set = {r['log_date'] for r in logs}
    streak = 0
    for i in range(60):
        if (today-timedelta(days=i)).isoformat() in date_set: streak+=1
        else: break
    if streak >= 3:  earned.add('streak_3')
    if streak >= 7:  earned.add('streak_7')
    if streak >= 30: earned.add('streak_30')

    # Low stress week
    recent = [r for r in logs if r['log_date'] >= (today-timedelta(days=6)).isoformat()]
    if len(recent) >= 5 and all(r['stress'] <= 2 for r in recent): earned.add('low_stress')

    # Rising mood
    moods = [r['mood'] for r in logs[-3:]] if len(logs)>=3 else []
    if len(moods)==3 and moods[0]<moods[1]<moods[2]: earned.add('mood_boost')

    return earned

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    uid  = session['user_id']
    conn = get_db()
    if request.method == 'POST':
        name       = request.form.get('name','').strip()[:80]
        course     = request.form.get('course','').strip()[:100]
        university = request.form.get('university','').strip()[:100]
        color      = request.form.get('avatar_color','#7C6EFA')
        new_pwd    = request.form.get('new_password','')
        cur_pwd    = request.form.get('current_password','')

        user = conn.execute('SELECT * FROM users WHERE id=?',(uid,)).fetchone()
        updates = []
        params  = []
        if name:       updates.append('name=?');       params.append(name)
        if course is not None: updates.append('course=?');     params.append(course)
        if university is not None: updates.append('university=?'); params.append(university)
        if color:      updates.append('avatar_color=?');params.append(color)

        if new_pwd:
            if check_password_hash(user['password'], cur_pwd):
                updates.append('password=?'); params.append(generate_password_hash(new_pwd))
            else:
                flash('Current password is incorrect.','error')
                conn.close()
                return redirect(url_for('profile'))

        if updates:
            params.append(uid)
            conn.execute(f"UPDATE users SET {','.join(updates)} WHERE id=?", params)
            conn.commit()
            session['user_name']  = name or session['user_name']
            session['user_color'] = color
            flash('Profile updated!','success')

    user   = conn.execute('SELECT * FROM users WHERE id=?',(uid,)).fetchone()
    earned = compute_badges(uid, conn)
    conn.close()

    all_badges = [dict(b, earned=(b['id'] in earned)) for b in BADGES]
    return render_template('profile.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        user=dict(user), badges=all_badges, earned_count=len(earned))

@app.route('/notifications')
@login_required
def notifications():
    uid  = session['user_id']
    conn = get_db()
    today_log = conn.execute(
        'SELECT id FROM daily_logs WHERE user_id=? AND log_date=CURRENT_DATE',(uid,)).fetchone()
    last_journal = conn.execute(
        'SELECT created_at FROM journal_entries WHERE user_id=? ORDER BY created_at DESC LIMIT 1',(uid,)).fetchone()
    active_goals = conn.execute(
        "SELECT COUNT(*) as c FROM goals WHERE user_id=? AND status='active'",(uid,)).fetchone()['c']
    today_goal_cis = conn.execute(
        'SELECT COUNT(*) as c FROM goal_checkins WHERE user_id=? AND checkin_date=CURRENT_DATE',(uid,)).fetchone()['c']
    earned = compute_badges(uid, conn)
    conn.close()

    notes = []
    if not today_log:
        notes.append({'type':'reminder','icon':'📅','title':'Daily Check-In Pending',
            'text':"You haven't logged today yet. It takes under 10 seconds!",'link':'/assessment','cta':'Check In Now'})
    if last_journal:
        from datetime import datetime as dt
        last = dt.strptime(last_journal['created_at'][:10], '%Y-%m-%d').date()
        gap  = (date.today() - last).days
        if gap >= 3:
            notes.append({'type':'reminder','icon':'📓','title':'Journal Reminder',
                'text':f"You haven't journaled in {gap} days. Writing helps process emotions.",'link':'/journal/new','cta':'Write Now'})
    if active_goals > 0 and today_goal_cis < active_goals:
        notes.append({'type':'reminder','icon':'🎯','title':'Goals Need Attention',
            'text':f"{active_goals - today_goal_cis} active goal(s) don't have today's check-in yet.",'link':'/goals','cta':'Check In'})
    if len(earned) >= 3:
        notes.append({'type':'achievement','icon':'🏆','title':'Achievements Unlocked',
            'text':f"You've earned {len(earned)} badges! Keep up the great work.",'link':'/profile','cta':'View Badges'})
    if not notes:
        notes.append({'type':'info','icon':'✅','title':"You're all caught up!",
            'text':'All reminders completed. Great job staying on top of your wellness.','link':'/dashboard','cta':'View Dashboard'})

    return render_template('notifications.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'),
        notifications=notes)

@app.route('/onboarding', methods=['GET','POST'])
@login_required
def onboarding():
    if request.method == 'POST':
        session['onboarded'] = True
        return redirect(url_for('assessment'))
    return render_template('onboarding.html',
        user_name=session['user_name'], user_color=session.get('user_color','#7C6EFA'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html',
        user_name=session.get('user_name',''),
        user_color=session.get('user_color','#7C6EFA')), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('404.html',
        user_name=session.get('user_name',''),
        user_color=session.get('user_color','#7C6EFA'),
        error_code=500, error_msg="Something went wrong on our end."), 500
