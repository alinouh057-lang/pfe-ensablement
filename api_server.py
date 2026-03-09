"""
========================================
PFE - Détection Ensablement Panneaux PV
API Flask - Version Cloud (Render.com)
========================================
Usage : gunicorn api_server:app
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import base64
from datetime import datetime
from collections import deque

app = Flask(__name__)
CORS(app)

# ===== CONFIG =====
MODEL_PATH   = 'models/final_model.pth'
IMG_SIZE     = 224
NUM_CLASSES  = 5
HISTORY_MAX  = 100
PHOTOS_DIR   = 'photos_recues'
SEUIL_ALERTE = 50

CLASSES = ['0_propre', '25_leger', '50_moyen', '75_fort', '100_complet']
LABELS  = ['0%', '25%', '50%', '75%', '100%']

os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== CHARGEMENT MODÈLE =====
model  = None
device = torch.device('cpu')

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Modèle non trouvé : {MODEL_PATH}")
        print("   Upload le modèle via /upload_model")
        return False

    print("🧠 Chargement ResNet50...")
    m = models.resnet50(weights=None)
    num_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.15),
        nn.Linear(256, NUM_CLASSES)
    )
    m.load_state_dict(torch.load(
        MODEL_PATH, weights_only=True, map_location=device))
    m.eval()
    model = m
    print("✅ Modèle chargé !")
    return True

load_model()

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== HISTORIQUE =====
history = deque(maxlen=HISTORY_MAX)

# ===== PRÉDICTION =====
def predict_image(image_bytes):
    img    = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)
        conf, pred  = probs.max(1)

    idx       = pred.item()
    taux      = int(LABELS[idx].replace('%', ''))
    all_probs = {LABELS[i]: round(probs[0][i].item() * 100, 2)
                 for i in range(NUM_CLASSES)}

    return {
        'taux'        : taux,
        'label'       : LABELS[idx],
        'classe'      : CLASSES[idx],
        'confiance'   : round(conf.item() * 100, 2),
        'probabilites': all_probs,
        'alerte'      : taux >= SEUIL_ALERTE,
        'message'     : get_message(taux)
    }

def get_message(taux):
    messages = {
        0  : "✅ Panneau propre - Aucune action nécessaire",
        25 : "🟡 Légèrement ensablé - Surveiller",
        50 : "🟠 Moyennement ensablé - Nettoyage recommandé",
        75 : "🔴 Fortement ensablé - Nettoyage urgent",
        100: "🚨 Complètement ensablé - Nettoyage immédiat !"
    }
    return messages.get(taux, "⚠️ Niveau inconnu")

# ===== ROUTES =====

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status'      : 'ok',
        'modele_pret' : model is not None,
        'modele'      : 'ResNet50',
        'accuracy'    : '85.94%',
        'predictions' : len(history),
        'timestamp'   : datetime.now().isoformat()
    })

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    Permet d'uploader le modèle .pth sur le serveur cloud
    Usage : curl -X POST -F "model=@final_model.pth" https://ton-app.onrender.com/upload_model
    """
    if 'model' not in request.files:
        return jsonify({'error': 'Fichier modèle manquant'}), 400

    file = request.files['model']
    os.makedirs('models', exist_ok=True)
    file.save(MODEL_PATH)

    success = load_model()
    if success:
        return jsonify({'status': 'ok', 'message': '✅ Modèle uploadé et chargé !'}), 200
    return jsonify({'error': 'Erreur chargement modèle'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Modèle non chargé. Upload via /upload_model'
        }), 503

    try:
        # Récupère image
        if 'image' in request.files:
            image_bytes = request.files['image'].read()
        elif request.content_type and 'image' in request.content_type:
            image_bytes = request.data
        elif request.is_json:
            data        = request.get_json()
            b64         = data.get('image', '').split(',')[-1]
            image_bytes = base64.b64decode(b64)
        else:
            image_bytes = request.data

        if not image_bytes:
            return jsonify({'error': 'Image vide'}), 400

        result    = predict_image(image_bytes)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sauvegarde photo
        photo_path = f"{PHOTOS_DIR}/photo_{timestamp}_{result['label']}.jpg"
        with open(photo_path, 'wb') as f:
            f.write(image_bytes)

        entry = {
            'timestamp' : timestamp,
            'datetime'  : datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            **result
        }
        history.appendleft(entry)

        print(f"[{timestamp}] {result['label']} ({result['confiance']:.1f}%) "
              f"| Alerte: {result['alerte']}")

        return jsonify(entry), 200

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'count': len(history), 'data': list(history)})

@app.route('/latest', methods=['GET'])
def get_latest():
    if history:
        return jsonify(history[0])
    return jsonify({'message': 'Aucune prédiction encore'}), 404

@app.route('/', methods=['GET'])
@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template_string(DASHBOARD_HTML)

# ===== DASHBOARD HTML =====
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PFE - Détection Ensablement Panneaux PV</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
  :root {
    --bg:#0a0e1a; --card:#111827; --border:#1e3a5f;
    --accent:#00d4ff; --green:#00ff88; --yellow:#ffcc00;
    --orange:#ff8800; --red:#ff3366; --text:#e2e8f0; --muted:#64748b;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Exo 2',sans-serif; min-height:100vh;
    background-image: radial-gradient(ellipse at 20% 50%,rgba(0,100,200,.08) 0%,transparent 60%),
    radial-gradient(ellipse at 80% 20%,rgba(0,212,255,.05) 0%,transparent 50%); }
  header { padding:20px 30px; border-bottom:1px solid var(--border);
    display:flex; justify-content:space-between; align-items:center;
    background:rgba(0,212,255,.03); }
  header h1 { font-size:1.2rem; font-weight:700; letter-spacing:2px;
    text-transform:uppercase; color:var(--accent); }
  .live { display:flex; align-items:center; gap:8px;
    font-family:'Share Tech Mono',monospace; font-size:.8rem; color:var(--green); }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--green);
    animation:pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.8)} }
  .container { padding:25px 30px; }
  .grid-top { display:grid; grid-template-columns:repeat(4,1fr); gap:15px; margin-bottom:20px; }
  .stat { background:var(--card); border:1px solid var(--border); border-radius:12px;
    padding:20px; text-align:center; position:relative; overflow:hidden; }
  .stat::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--accent); }
  .stat-val { font-size:2rem; font-weight:700; font-family:'Share Tech Mono',monospace; color:var(--accent); }
  .stat-lbl { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-top:5px; }
  .grid-main { display:grid; grid-template-columns:1fr 1.8fr; gap:20px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }
  .card-title { font-size:.72rem; text-transform:uppercase; letter-spacing:2px;
    color:var(--muted); margin-bottom:15px; }
  .taux-display { text-align:center; padding:20px 0; }
  .taux-num { font-size:5rem; font-weight:700; font-family:'Share Tech Mono',monospace; line-height:1; }
  .taux-lbl { font-size:.9rem; color:var(--muted); margin-top:5px; text-transform:uppercase; letter-spacing:2px; }
  .gauge-bg { height:12px; background:rgba(255,255,255,.05); border-radius:6px; margin:20px 0; overflow:hidden; }
  .gauge-fill { height:100%; border-radius:6px; transition:width .8s ease; }
  .msg-box { padding:12px 16px; border-radius:8px; font-size:.88rem; text-align:center;
    margin-top:15px; font-weight:600; }
  .prob-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; font-size:.85rem; }
  .prob-lbl { width:35px; font-family:'Share Tech Mono',monospace; color:var(--muted); }
  .prob-bg { flex:1; height:8px; background:rgba(255,255,255,.05); border-radius:4px; overflow:hidden; }
  .prob-bar { height:100%; border-radius:4px; background:var(--accent); transition:width .6s ease; }
  .prob-val { width:48px; text-align:right; font-family:'Share Tech Mono',monospace; font-size:.78rem; }
  table { width:100%; border-collapse:collapse; font-size:.82rem; margin-top:10px; }
  th { text-align:left; color:var(--muted); font-size:.7rem; text-transform:uppercase;
    letter-spacing:1px; padding:8px 10px; border-bottom:1px solid var(--border); }
  td { padding:10px; border-bottom:1px solid rgba(30,58,95,.4); font-family:'Share Tech Mono',monospace; }
  .badge { padding:3px 10px; border-radius:20px; font-size:.72rem; font-weight:600; }
  .ok  { background:rgba(0,255,136,.1);  color:var(--green); }
  .bad { background:rgba(255,51,102,.2); color:var(--red); }
  footer { text-align:center; padding:15px; color:var(--muted); font-size:.72rem;
    border-top:1px solid var(--border); font-family:'Share Tech Mono',monospace; margin-top:20px; }
</style>
</head>
<body>
<header>
  <h1>☀️ PFE · Détection Ensablement Panneaux PV</h1>
  <div class="live"><div class="dot"></div>LIVE · Auto-refresh 5s</div>
</header>
<div class="container">
  <div class="grid-top">
    <div class="stat"><div class="stat-val" id="s-taux">--</div><div class="stat-lbl">Taux Actuel</div></div>
    <div class="stat"><div class="stat-val" id="s-conf">--</div><div class="stat-lbl">Confiance IA</div></div>
    <div class="stat"><div class="stat-val" id="s-total">0</div><div class="stat-lbl">Total Analyses</div></div>
    <div class="stat"><div class="stat-val" id="s-alerte" style="color:var(--red)">0</div><div class="stat-lbl">Alertes</div></div>
  </div>
  <div class="grid-main">
    <div>
      <div class="card" style="margin-bottom:15px">
        <div class="card-title">📊 Taux d'Ensablement</div>
        <div class="taux-display">
          <div class="taux-num" id="taux-num" style="color:var(--green)">--</div>
          <div class="taux-lbl" id="taux-lbl">En attente ESP32-CAM...</div>
        </div>
        <div class="gauge-bg"><div class="gauge-fill" id="gauge" style="width:0%;background:var(--green)"></div></div>
        <div class="msg-box" id="msg" style="background:rgba(0,255,136,.08);color:var(--green)">
          En attente de données...
        </div>
      </div>
      <div class="card">
        <div class="card-title">🎯 Probabilités par classe</div>
        <div id="probs">
          <div class="prob-row"><span class="prob-lbl">0%</span><div class="prob-bg"><div class="prob-bar" style="width:0%"></div></div><span class="prob-val">0%</span></div>
          <div class="prob-row"><span class="prob-lbl">25%</span><div class="prob-bg"><div class="prob-bar" style="width:0%"></div></div><span class="prob-val">0%</span></div>
          <div class="prob-row"><span class="prob-lbl">50%</span><div class="prob-bg"><div class="prob-bar" style="width:0%"></div></div><span class="prob-val">0%</span></div>
          <div class="prob-row"><span class="prob-lbl">75%</span><div class="prob-bg"><div class="prob-bar" style="width:0%"></div></div><span class="prob-val">0%</span></div>
          <div class="prob-row"><span class="prob-lbl">100%</span><div class="prob-bg"><div class="prob-bar" style="width:0%"></div></div><span class="prob-val">0%</span></div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">📋 Historique des Analyses</div>
      <table><thead><tr><th>Date/Heure</th><th>Taux</th><th>Confiance</th><th>Statut</th></tr></thead>
      <tbody id="hist"><tr><td colspan="4" style="text-align:center;color:var(--muted);padding:30px">
        En attente de la première analyse...
      </td></tr></tbody></table>
    </div>
  </div>
</div>
<footer>PFE · Détection Intelligente Ensablement · Panneaux PV · ResNet50 · 85.94% Accuracy</footer>
<script>
  const C = t => t===0?'var(--green)':t<=25?'var(--yellow)':t<=50?'var(--orange)':'var(--red)';
  async function update() {
    try {
      const r1 = await fetch('/latest');
      if(r1.ok){
        const d = await r1.json();
        const c = C(d.taux);
        document.getElementById('taux-num').textContent = d.label;
        document.getElementById('taux-num').style.color = c;
        document.getElementById('taux-lbl').textContent = d.classe.replace('_',' ').toUpperCase();
        document.getElementById('s-taux').textContent   = d.label;
        document.getElementById('s-taux').style.color   = c;
        document.getElementById('s-conf').textContent   = d.confiance.toFixed(1)+'%';
        const g = document.getElementById('gauge');
        g.style.width = d.taux+'%'; g.style.background = c;
        const m = document.getElementById('msg');
        m.textContent = d.message; m.style.color = c;
        m.style.background = d.alerte?'rgba(255,51,102,.12)':'rgba(0,255,136,.07)';
        const rows = document.getElementById('probs').querySelectorAll('.prob-row');
        ['0%','25%','50%','75%','100%'].forEach((l,i)=>{
          const v = d.probabilites[l]||0;
          rows[i].querySelector('.prob-bar').style.width = v+'%';
          rows[i].querySelector('.prob-val').textContent = v.toFixed(1)+'%';
        });
      }
      const r2 = await fetch('/history');
      if(r2.ok){
        const data = await r2.json();
        document.getElementById('s-total').textContent  = data.count;
        document.getElementById('s-alerte').textContent = data.data.filter(d=>d.alerte).length;
        if(data.data.length>0){
          document.getElementById('hist').innerHTML = data.data.slice(0,15).map(d=>`
            <tr>
              <td>${d.datetime}</td>
              <td style="color:${C(d.taux)};font-weight:600">${d.label}</td>
              <td>${d.confiance.toFixed(1)}%</td>
              <td><span class="badge ${d.alerte?'bad':'ok'}">${d.alerte?'⚠️ ALERTE':'✅ OK'}</span></td>
            </tr>`).join('');
        }
      }
    } catch(e){ console.log(e); }
  }
  update();
  setInterval(update, 5000);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 API démarrée sur port {port}")
    print(f"   Dashboard : http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
