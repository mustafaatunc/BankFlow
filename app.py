import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import os
from dotenv import load_dotenv
load_dotenv()
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import sqlite3
import bcrypt
import hashlib
from streamlit_option_menu import option_menu
import datetime
from xai_engine import explain_prediction

# --- 1. SAYFA AYARLARI VE CSS ---
st.set_page_config(page_title="BankFlow | Kurumsal Kredi Yönetimi", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .metric-card { 
        background-color: #1f2937; padding: 20px; border-radius: 10px; border: 1px solid #374151; 
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #2563eb; color: white; border: none; }
    .stButton>button:hover { background-color: #1d4ed8; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# --- 2. VERİ TABANI VE GÜVENLİK ---
def init_db():
    with sqlite3.connect('banka_veritabani.db') as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT, role TEXT, name TEXT)')
        c.execute('''CREATE TABLE IF NOT EXISTS credit_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            masked_tc TEXT, tc_hash TEXT, musteri_yas INTEGER, kredi_miktari INTEGER, 
            vade INTEGER, risk_skoru INTEGER, sonuc TEXT, durum TEXT, personel TEXT, tarih TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value REAL)')
        c.execute(
            'CREATE TABLE IF NOT EXISTS audit_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, action TEXT, details TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')

        c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('risk_threshold', 1400)")

        admin_mail = 'admin@admin.com'
        admin_check = c.execute("SELECT * FROM users WHERE email=?", (admin_mail,)).fetchone()
        if not admin_check:
            sifre = os.getenv('ADMIN_PASSWORD')
            if sifre: 
                hashed_admin_pass = bcrypt.hashpw(sifre.encode(), bcrypt.gensalt()).decode()
                c.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (admin_mail, hashed_admin_pass, 'admin', 'Şube Müdürü'))

def get_tc_hash(tc):
    return hashlib.sha256(tc.encode()).hexdigest()


def mask_tc(tc):
    return f"{tc[:3]}*****{tc[-3:]}"


def log_action(user, action, details=""):
    with sqlite3.connect('banka_veritabani.db') as conn:
        conn.execute("INSERT INTO audit_logs (user, action, details) VALUES (?, ?, ?)", (user, action, str(details)))


def get_db_data(query, params=()):
    with sqlite3.connect('banka_veritabani.db') as conn:
        return pd.read_sql_query(query, conn, params=params)


def execute_db(query, params=()):
    with sqlite3.connect('banka_veritabani.db') as conn:
        conn.execute(query, params)


def add_history(tc, yas, miktar, vade, skor, sonuc, durum, personel):
    masked = mask_tc(tc)
    h_tc = get_tc_hash(tc)
    with sqlite3.connect('banka_veritabani.db') as conn:
        conn.execute('''INSERT INTO credit_history 
            (masked_tc, tc_hash, musteri_yas, kredi_miktari, vade, risk_skoru, sonuc, durum, personel) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (masked, h_tc, yas, miktar, vade, skor, sonuc, durum, personel))


# --- 3. MODEL YÜKLEME ---
@st.cache_resource
def load_assets():
    try:
        return tf.keras.models.load_model('kredi_risk_modeli.keras'), joblib.load('veri_isleyici.pkl')
    except:
        return None, None


model, preprocessor = load_assets()
init_db()


# --- 4. MODERN HİBRİT KARAR MOTORU ---
def calculate_hybrid_score(raw_score, inputs):
    score = raw_score
    msgs = []
    scaled_amt = inputs['credit_amount']
    job_code = inputs['job']
    history = inputs['credit_history']
    housing = inputs['housing']
    age = inputs['age']
    install_rate = inputs['installment_rate']

    if job_code == 'A171' and history == 'A34':
        if scaled_amt > 10000:
            score += 750;
            msgs.append("🌟 VIP Segment: Kurumsal onay desteği (+750)")
        else:
            score += 300;
            msgs.append("✅ Gelir Gücü: Yönetici statüsü bonusu (+300)")
    elif job_code == 'A173':
        score += 150;
        msgs.append("✅ İstihdam: Nitelikli personel bonusu (+150)")

    if history == 'A34':
        score += 250;
        msgs.append("✅ Finansal Sicil: Kusursuz ödeme geçmişi (+250)")
    elif history in ['A30', 'A31', 'A33']:
        score -= 450;
        msgs.append("⛔ Kritik Risk: KKB kayıtları sorunlu (-450)")

    if housing == 'A152': score += 150; msgs.append("✅ Teminat: Gayrimenkul güvencesi (+150)")
    if install_rate == 4: score -= 250; msgs.append("⛔ Borçlanma Oranı: Gelire göre taksitler çok yüksek (-250)")

    return int(max(0, min(1900, score))), msgs


def calculate_payment(amount, duration, interest):
    r = (interest / 100) / 12
    p = amount * (r * (1 + r) ** duration) / ((1 + r) ** duration - 1)
    return round(p, 2), round(p * duration, 2)


def clean_text(text):
    if text is None: return ""
    # Türkçe karakter eşleşmeleri
    repl = {'İ': 'I', 'ı': 'i', 'Ö': 'O', 'ö': 'o', 'Ü': 'U', 'ü': 'u', 'Ç': 'C', 'ç': 'c', 'Ğ': 'G', 'ğ': 'g',
            'Ş': 'S', 'ş': 's'}
    t = str(text)
    for k, v in repl.items():
        t = t.replace(k, v)

    # Emojileri ve latin-1 dışı karakterleri temizle
    # 'ignore' parametresi kodlanamayan karakterleri siler
    return t.encode('latin-1', 'ignore').decode('latin-1')


def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()

    # --- Başlık Bölümü ---
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 15, txt="BankFlow | Kredi Analiz ve Risk Raporu", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, txt=f"Rapor Tarihi: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    # TÜM teknik terimleri içeren mapping sözlüğü
    f_map = {
        "TC": "Musteri Kimlik (TCKN)",
        "Skor": "Kredi Risk Skoru",
        "Karar": "Tahsis Karari",
        "age": "Yas",
        "credit_amount": "Kredi Tutari",
        "duration": "Vade (Ay)",
        "installment_rate": "Taksit/Gelir Orani",
        "job": "Meslek Grubu",
        "housing": "Konut Durumu",
        "credit_history": "Kredi Gecmisi (KKB)",
        "Kredi Tutarı": "Kredi Tutari",
        "Vade": "Vade"
    }

    def draw_section(title):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(243, 244, 246)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"  {clean_text(title)}", ln=True, fill=True)
        pdf.ln(3)

    # --- BÖLÜM 1: Müşteri Bilgileri (Tek Döngü) ---
    draw_section("MUSTERI VE BASVURU OZETI")
    pdf.set_font("Arial", size=11)

    # Bilgileri sıralı ve tek seferde yazdırıyoruz
    for k, v in data.items():
        if k not in ['xai', 'msgs']:
            label = f_map.get(k, k)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(45, 8, txt=f"{clean_text(label)}:", ln=0)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 8, txt=f"{clean_text(v)}", ln=True)
    pdf.ln(5)

    # --- BÖLÜM 2: Banka Notları ---
    if 'msgs' in data and data['msgs']:
        draw_section("BANKA POLITIKASI VE NOTLAR")
        pdf.set_font("Arial", 'I', 10)
        for msg in data['msgs']:
            if msg:  # None kontrolü burada da önemli
                pdf.multi_cell(0, 7, txt=f"- {clean_text(msg)}")
        pdf.ln(5)

    # --- BÖLÜM 3: XAI Tablosu (İngilizce Terimleri Türkçeleştirme) ---
    if 'xai' in data:
        draw_section("YAPAY ZEKA FAKTOR ANALIZI")
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(70, 8, "Kriter", ln=0)
        pdf.cell(60, 8, "Skora Etkisi", ln=1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())

        for e in data['xai']:
            # e['feature'] içindeki 'credit_history' gibi terimleri f_map ile Türkçeye çevir
            feat_name = f_map.get(e['feature'], e['feature'])
            pdf.set_font("Arial", size=10)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(70, 8, clean_text(feat_name), ln=0)

            delta_val = e['delta']
            if delta_val < 0:
                pdf.set_text_color(220, 38, 38)  # Kırmızı
            else:
                pdf.set_text_color(22, 163, 74)  # Yeşil
            pdf.cell(60, 8, f"{'+' if delta_val > 0 else ''}{delta_val} Puan", ln=1)

    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, txt="Bu rapor BankFlow tarafindan uretilmistir.", align='C')

    return pdf.output(dest='S').encode('latin-1')


maps = {
    'checking_account': {'Mevcut Hesap Yok (Güvenli)': 'A14', 'Eksi Bakiye (Riskli)': 'A11', 'Düşük Bakiye': 'A12',
                         'Yüksek Bakiye': 'A13'},
    'credit_history': {'Kusursuz (Düzenli)': 'A34', 'İyi (Sorunsuz)': 'A32', 'Orta': 'A31', 'Zayıf': 'A33',
                       'Kritik': 'A30'},
    'purpose': {'Ticari': 'A49', 'Yeni Araç': 'A40', 'İkinci El': 'A41', 'Eşya': 'A42', 'Tadilat': 'A48',
                'Eğitim': 'A46', 'Diğer': 'A410'},
    'savings_account': {'Yok / Bilinmiyor': 'A65', 'Düşük': 'A61', 'Orta': 'A62', 'Yüksek': 'A63', 'Çok Yüksek': 'A64'},
    'employment': {'İşsiz': 'A71', '< 1 Yıl': 'A72', '1 - 4 Yıl': 'A73', '4 - 7 Yıl': 'A74', '> 7 Yıl': 'A75'},
    'status_sex': {'Erkek (Evli)': 'A94', 'Erkek (Bekar)': 'A93', 'Kadın': 'A92'},
    'housing': {'Ev Sahibi': 'A152', 'Kiracı': 'A151', 'Lojman': 'A153'},
    'job': {'Yönetici/İşveren': 'A171', 'Uzman/Nitelikli': 'A173', 'Vasıfsız Yerleşik': 'A172',
            'Vasıfsız Geçici': 'A174'},
    'property': {'Gayrimenkul': 'A121', 'Araç Ruhsatı': 'A123', 'Sigorta': 'A122', 'Yok': 'A124'},
    'telephone': {'Var': 'A192', 'Yok': 'A191'}
}

# --- 5. GİRİŞ VE PANEL ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
        st.title("BankFlow Portal")
        l_tab, s_tab = st.tabs(["🔐 Personel Girişi", "🔑 Şifremi Belirle"])

        with l_tab:
            email = st.text_input("Kurumsal E-posta", key="login_email")
            password = st.text_input("Şifre", type='password', key="login_pass")
            if st.button("GİRİŞ YAP"):
                user = get_db_data("SELECT * FROM users WHERE email=?", (email,))
                if not user.empty:
                    stored_password = user.iloc[0]['password']
                    # Çift else hatası giderildi
                    if stored_password and bcrypt.checkpw(password.encode(), stored_password.encode()):
                        st.session_state.clear()
                        st.session_state.update({'logged_in': True, 'email': email, 'role': user.iloc[0]['role'],
                                                 'name': user.iloc[0]['name']})
                        log_action(email, "Giriş Yapıldı")
                        st.rerun()
                    else:
                        st.error("Hatalı şifre!")
                else:
                    st.error("Kullanıcı bulunamadı!")

        with s_tab:
            st.info("Müdürünüz hesabınızı oluşturduysa buradan şifre belirleyebilirsiniz.")
            s_mail = st.text_input("Kurumsal E-posta", key="s_email")
            s_p1 = st.text_input("Yeni Şifre", type='password', key="s_p1")
            s_p2 = st.text_input("Tekrar Şifre", type='password', key="s_p2")
            if st.button("ŞİFREYİ KAYDET"):
                if s_p1 != s_p2:
                    st.error("Şifreler uyuşmuyor!")
                else:
                    u_chk = get_db_data("SELECT * FROM users WHERE email=?", (s_mail,))
                    if u_chk.empty:
                        st.error("E-posta sistemde tanımlı değil!")
                    else:
                        hashed_p = bcrypt.hashpw(s_p1.encode(), bcrypt.gensalt()).decode()
                        execute_db("UPDATE users SET password=? WHERE email=?", (hashed_p, s_mail))
                        st.success("Şifre güncellendi.")
                        time.sleep(1);
                        st.rerun()
else:
    with st.sidebar:
        st.write(f"### 👤 {st.session_state['name']}")
        if st.session_state['role'] == 'admin':
            pending_count = len(get_db_data("SELECT id FROM credit_history WHERE durum='MÜDÜR ONAYINDA'"))
            if pending_count > 0: st.sidebar.error(f"🔔 {pending_count} Dosya Onay Bekliyor!")

        if st.session_state['role'] == 'admin':
            m_opts = ["📈 Genel Performans","📂 Toplu Sorgulama", "👥 Personel Yönetimi", "⚙️ Banka Politikası", "🛡️ Hareketler", "Çıkış"]
            m_icons = ["bar-chart-fill", "file-earmark-spreadsheet-fill", "people-fill", "gear-fill",
                       "shield-lock-fill", "box-arrow-right"]
        else:
            m_opts = ["📝 Kredi Başvurusu","📋 Başvurularım", "Çıkış"]
            m_icons = ["pencil-square", "list-task", "box-arrow-right"]
        sel = option_menu("Banka Menü", m_opts, icons=m_icons, menu_icon="bank", default_index=0)
        if sel == "Çıkış":
            st.session_state.clear()  # Tüm oturum verilerini (analiz sonuçları dahil) siler
            st.rerun()

    if sel == "📈 Genel Performans":
        st.title("📊 Şube ve Personel Verimlilik Analizi")
        df = get_db_data("""
            SELECT ch.*, u.role 
            FROM credit_history ch 
            LEFT JOIN users u ON ch.personel = u.name
        """)

        if st.session_state['role'] == 'admin':
            st.divider()
            st.subheader("⚠️ Karar Bekleyen Yüksek Tutarlı Başvurular")

            pending_query = "SELECT id, masked_tc, kredi_miktari, vade, risk_skoru, personel, tarih FROM credit_history WHERE durum='MÜDÜR ONAYINDA'"
            pending_df = get_db_data(pending_query)

            if not pending_df.empty:
                for _, row in pending_df.iterrows():
                    # Her dosya için şık bir kutu (expander) oluşturuyoruz
                    with st.expander(
                            f"📁 Dosya No: {row['id']} | Müşteri: {row['masked_tc']} | {row['kredi_miktari']:,} TL"):
                        c1, c2, c3 = st.columns(3)
                        c1.write(f"**Personel:** {row['personel']}")
                        c2.write(f"**Risk Skoru:** {row['risk_skoru']}")
                        c3.write(f"**Tarih:** {row['tarih']}")

                        # Onay ve Red Butonları
                        btn_onay, btn_red = st.columns(2)

                        if btn_onay.button(f"✅ ONAYLA (ID: {row['id']})", use_container_width=True):
                            execute_db("UPDATE credit_history SET sonuc='ONAYLANDI', durum='TAMAMLANDI' WHERE id=?",
                                       (row['id'],))
                            log_action(st.session_state['email'], "Müdür Onayı Verildi", f"Dosya ID: {row['id']}")
                            st.success(f"Dosya {row['id']} onaylandı.")
                            time.sleep(1)
                            st.rerun()  # Sayfayı yenileyerek listeyi günceller

                        if btn_red.button(f"❌ REDDET (ID: {row['id']})", use_container_width=True):
                            execute_db("UPDATE credit_history SET sonuc='REDDEDİLDİ', durum='TAMAMLANDI' WHERE id=?",
                                       (row['id'],))
                            log_action(st.session_state['email'], "Müdür Reddi Verildi", f"Dosya ID: {row['id']}")
                            st.warning(f"Dosya {row['id']} reddedildi.")
                            time.sleep(1)
                            st.rerun()
            else:
                st.success("✅ Onay bekleyen herhangi bir dosya bulunmuyor.")

        if not df.empty:
            df['Durum'] = df['sonuc'].apply(lambda x: 'Onay' if 'ONAY' in x else 'Red')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Sorgu", len(df))
            c2.metric("Onaylanan Hacim", f"{df[df['Durum'] == 'Onay']['kredi_miktari'].sum():,.0f} TL")
            c3.metric("Onay Oranı", f"%{(len(df[df['Durum'] == 'Onay']) / len(df)) * 100:.1f}")
            c4.metric("Ortalama Risk Skoru", int(df['risk_skoru'].mean()))

            st.divider();
            st.subheader("🏆 Personel Performans Analizi")
            perf = df.groupby(['personel', 'Durum']).size().unstack(fill_value=0)
            if 'Onay' not in perf: perf['Onay'] = 0
            if 'Red' not in perf: perf['Red'] = 0
            st.table(perf.rename(columns={'Onay': '✅ Onaylanan Adet', 'Red': '❌ Reddedilen Adet'}))

            t1, t2, t3 = st.tabs(["📊 Hacim Grafiği", "✅ Onaylananlar", "❌ Reddedilenler"])
            with t1:
                if not df.empty:
                    # Rol isimlerini daha şık hale getirelim
                    df['rol_etiket'] = df['role'].map({'admin': '🏦 ŞUBE MÜDÜRÜ', 'personel': '👥 PERSONEL'})

                    # Grafik oluşturma
                    fig = px.bar(df,
                                 x='personel',
                                 y='kredi_miktari',
                                 color='Durum',
                                 barmode='group',
                                 facet_col='rol_etiket',  # Müdür ve personeli ayrı bölmelere ayırır
                                 pattern_shape='rol_etiket',  # Müdür barlarına ayırıcı bir desen ekler
                                 category_orders={"rol_etiket": ["🏦 ŞUBE MÜDÜRÜ", "👥 PERSONEL"]},
                                 labels={
                                     'personel': 'Yetkili',
                                     'kredi_miktari': 'Kredi Hacmi (TL)',
                                     'Durum': 'Karar'
                                 },
                                 color_discrete_map={'Onay': '#22c55e', 'Red': '#ef4444'})

                    fig.update_layout(
                        xaxis_title="",
                        yaxis_title="Toplam Hacim (TL)",
                        showlegend=True
                    )

                    # Grafik başlıklarındaki "rol_etiket=" yazısını temizleyip sadece başlığı bırakır
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

                    st.plotly_chart(fig, use_container_width=True)
            with t2:
                st.dataframe(df[df['Durum'] == 'Onay'].drop(columns=['tc_hash']), use_container_width=True)
            with t3:
                st.dataframe(df[df['Durum'] == 'Red'].drop(columns=['tc_hash']), use_container_width=True)
        else:
            st.info("Sistemde henüz kayıtlı veri bulunmuyor.")

    elif sel == "👥 Personel Yönetimi":
        st.title("👥 Kullanıcı ve Personel Yönetimi")
        tab1, tab2 = st.tabs(["Personel Listesi", "Yeni Personel Ekle"])
        with tab1:
            u_df = get_db_data("SELECT name, email, role FROM users")
            st.dataframe(u_df, use_container_width=True)
            d_mail = st.selectbox("Silinecek Hesap", u_df['email'].tolist())
            if st.button("Sistemden Sil"):
                if d_mail == 'admin@admin.com':
                    st.error("Müdür silinemez!")
                else:
                    execute_db("DELETE FROM users WHERE email=?", (d_mail,))
                    log_action(st.session_state['email'], "Personel Silindi", d_mail);
                    st.success("Silindi.");
                    st.rerun()
        with tab2:
            with st.form("add"):
                n_name = st.text_input("Ad Soyad")
                n_mail = st.text_input("Kurumsal E-posta")
                n_role = st.selectbox("Yetki", ["personel", "yönetici"])
                if st.form_submit_button("Personeli Tanımla"):
                    execute_db("INSERT INTO users (email, password, role, name) VALUES (?,?,?,?)",
                               (n_mail, None, n_role, n_name))
                    st.success("Tanımlandı!");
                    st.rerun()

    elif sel == "⚙️ Banka Politikası":
        st.title("⚙️ Kredi Risk Politikası Ayarları")
        curr = get_db_data("SELECT value FROM settings WHERE key='risk_threshold'").iloc[0]['value']
        new_thr = st.slider("Yeni Eşik Değeri", 1000, 1800, int(curr), step=10)
        if st.button("Politikayı Güncelle"):
            execute_db("UPDATE settings SET value=? WHERE key='risk_threshold'", (new_thr,))
            st.success("Güncellendi!");
            st.rerun()

    elif sel == "🛡️ Hareketler":
        st.title("🛡️ Güvenlik ve Denetim Kayıtları")
        st.dataframe(get_db_data("SELECT * FROM audit_logs ORDER BY timestamp DESC"), use_container_width=True)

    elif sel == "📝 Kredi Başvurusu":
        st.title("📝 Kredi Tahsis Ekranı")

        # Durum takibi için session_state kontrolleri
        if 'tc_verified' not in st.session_state: st.session_state['tc_verified'] = False
        if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None

        # 1. ADIM: T.C. SORGULAMA EKRANI
        if st.session_state['tc_verified'] is False:
            tc_c1, tc_c2 = st.columns([2, 1])
            in_tc = tc_c1.text_input("Müşteri T.C. Kimlik Numarası", max_chars=11)

            if tc_c2.button("Müşteri Sorgula"):
                if len(in_tc) == 11 and in_tc.isdigit():
                    h_tc = get_tc_hash(in_tc)
                    today = datetime.datetime.now().strftime('%Y-%m-%d')
                    check = get_db_data("SELECT id FROM credit_history WHERE tc_hash=? AND tarih LIKE ?",
                                        (h_tc, f"{today}%"))
                    if not check.empty:
                        st.error("⛔ Sorgu Sınırı: Bu müşteri için bugün zaten sorgulama yapılmış.")
                    else:
                        st.session_state.update({'tc_verified': True, 'active_tc': in_tc, 'analysis_result': None})
                        st.rerun()
                else:
                    st.error("Geçersiz TCKN!")

        # 2. ADIM: ANALİZ FORMU (Sadece TC doğrulandıysa görünür)
        elif st.session_state['tc_verified'] == True:
            st.success(f"Aktif Müşteri: {mask_tc(st.session_state['active_tc'])}")
            with st.form("credit"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    amt = st.number_input("Tutar (TL)", 5000, 10000000, 100000)
                    dur = st.slider("Vade (Ay)", 3, 120, 24)
                    job = st.selectbox("Meslek", list(maps['job'].keys()))
                with c2:
                    age = st.number_input("Yaş", 18, 90, 30)
                    check = st.selectbox("Mevduat", list(maps['checking_account'].keys()))
                    hist = st.selectbox("KKB Geçmişi", list(maps['credit_history'].keys()))
                with c3:
                    intr = st.number_input("Faiz (%)", 1.0, 10.0, 3.99)
                    sav = st.selectbox("Birikim", list(maps['savings_account'].keys()))
                    prop = st.selectbox("Teminat", list(maps['property'].keys()))

                with st.expander("Ek Detaylar"):
                    purp = st.selectbox("Amaç", list(maps['purpose'].keys()))
                    emp = st.selectbox("Kıdem", list(maps['employment'].keys()))
                    hs = st.selectbox("Konut", list(maps['housing'].keys()))
                    rate = st.slider("Borçlanma Oranı", 1, 4, 2)

                if st.form_submit_button("ANALİZİ TAMAMLA ✨"):
                    try:
                        MODEL_SCALE_FACTOR = 80
                        scaled_amt_for_ai = amt / MODEL_SCALE_FACTOR
                        inp = {'checking_account': maps['checking_account'][check], 'duration': dur,
                               'credit_history': maps['credit_history'][hist], 'purpose': maps['purpose'][purp],
                               'credit_amount': scaled_amt_for_ai, 'savings_account': maps['savings_account'][sav],
                               'employment': maps['employment'][emp], 'installment_rate': rate, 'status_sex': 'A93',
                               'guarantors': 'A101', 'residence_since': 4, 'property': maps['property'][prop],
                               'age': age, 'other_installments': 'A143', 'housing': maps['housing'][hs],
                               'existing_credits': 1, 'job': maps['job'][job], 'people_liable': 1, 'telephone': 'A192',
                               'foreign_worker': 'A201'}

                        proc = preprocessor.transform(pd.DataFrame([inp]))
                        risk = model.predict(proc, verbose=0)[0][0]
                        f, msgs = calculate_hybrid_score(int((1 - risk) * 1900), inp)
                        xai_res = explain_prediction(model, preprocessor, inp)
                        thr = get_db_data("SELECT value FROM settings WHERE key='risk_threshold'").iloc[0]['value']

                        kredi_durumu = "TAMAMLANDI"
                        if amt > 500000:
                            kredi_durumu = "MÜDÜR ONAYINDA"
                            dec, col = "MÜDÜR ONAYI BEKLİYOR", "#eab308"
                        else:
                            dec, col = ("ONAYLANABILIR", "#22c55e") if f >= thr else (
                                ("DEGERLENDIRILMELI", "#eab308") if f >= thr - 400 else ("RED ONERILIR", "#ef4444"))

                        mp, tp = calculate_payment(amt, dur, intr)
                        add_history(st.session_state['active_tc'], age, amt, dur, f, dec, kredi_durumu,
                                    st.session_state['name'])

                        st.session_state['analysis_result'] = {
                            'score': f,
                            'dec': dec,
                            'color': col,
                            'msgs': msgs,
                            'mp': mp,
                            'tp': tp,
                            'xai': xai_res,
                            'amt': amt,
                            'dur': dur
                        }
                        st.session_state['tc_verified'] = "DONE"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Hata: {e}")

        # 3. ADIM: SONUÇ PANELİ (Analiz bittiğinde 'DONE' durumunda görünür)
        elif st.session_state['tc_verified'] == "DONE":
            res = st.session_state['analysis_result']
            if res:
                st.success("🎯 Analiz Sonuçları Raporlandı")
                r1, r2 = st.columns([1, 2])
                with r1:
                    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=res['score'],
                                                           gauge={'axis': {'range': [0, 1900]},
                                                                  'bar': {'color': res['color']}})),
                                    use_container_width=True)
                with r2:
                    st.markdown(f"<h2 style='color:{res['color']};'>{res['dec']}</h2>", unsafe_allow_html=True)
                    for m in res['msgs']:
                        if m and str(m).strip().lower() != "none":
                            st.info(m)
                    st.info(f"💰 Taksit: {res['mp']} TL | Toplam: {res['tp']} TL")

                    st.subheader("🧠 Karar Açıklaması")
                    st.caption(
                        "🔍 Bu grafik, müşteri özelliklerindeki değişimlerin skoru nasıl etkileyeceğini gösterir.")
                    feature_tr = {
                        "age": "Müşteri Yaşı",
                        "credit_amount": "Kredi Tutarı",
                        "duration": "Vade Süresi (Ay)",
                        "installment_rate": "Taksit/Gelir Oranı",
                        "credit_history": "Kredi Geçmişi (KKB)",
                        "job": "Meslek Grubu",
                        "housing": "Konut Durumu"
                    }
                    xai_df = pd.DataFrame(res['xai']['effects'])
                    # İngilizce isimleri Türkçe karşılıklarıyla değiştiriyoruz
                    xai_df['feature'] = xai_df['feature'].map(lambda x: feature_tr.get(x, x))

                    fig_xai = px.bar(xai_df,
                                     x='delta',
                                     y='feature',
                                     orientation='h',
                                     color='delta',
                                     # Eksen etiketlerini Türkçeleştir
                                     labels={'delta': 'Skora Etkisi (Puan)', 'feature': 'Değerlendirilen Kriter'},
                                     color_continuous_scale=['#ef4444', '#22c55e'])

                    fig_xai.update_layout(showlegend=False)
                    st.plotly_chart(fig_xai, use_container_width=True)

                res = st.session_state['analysis_result']
                if res:
                    pdf_data = {
                        "TC": mask_tc(st.session_state['active_tc']),
                        "Skor": str(res['score']),
                        "Karar": res['dec'],
                        "Kredi Tutarı": f"{res.get('amt', 0):,} TL",  # Artık 0 gelmeyecek
                        "Vade": f"{res.get('dur', 0)} Ay",
                        "msgs": res.get('msgs', []),
                        "xai": res['xai']['effects']
                    }

                    st.download_button(
                        label="📄 Analiz Raporunu İndir",
                        data=create_pdf(pdf_data),  # create_pdf fonksiyonun clean_text içermeli
                        file_name=f"Kredi_Raporu_{st.session_state['active_tc']}.pdf",
                        mime="application/pdf"
                    )

                # 'with r2' bloğu dışına ama 'if res' içine yerleştirildi
                if st.button("🆕 Yeni Müşteri Sorgula"):
                    st.session_state.update({'tc_verified': False, 'analysis_result': None})
                    st.rerun()

    elif sel == "📋 Başvurularım":
        st.title("📋 Yaptığım Başvurular ve Güncel Durumlar")
        # Sadece giriş yapan personelin ismine göre filtreleme yapıyoruz
        my_tasks = get_db_data(
            "SELECT masked_tc, kredi_miktari, vade, risk_skoru, sonuc, durum, tarih FROM credit_history WHERE personel=? ORDER BY tarih DESC",
            (st.session_state['name'],))

        if not my_tasks.empty:
            st.dataframe(my_tasks, use_container_width=True)
        else:
            st.info("Henüz bir kredi başvurusu yapmadınız.")


    elif sel == "📂 Toplu Sorgulama":
        st.title("📂 Toplu Kredi Sorgulama")
        
        up = st.file_uploader("Analiz edilecek Excel listesini seçin", type=['xlsx'])

        if up:
            df_b = pd.read_excel(up)

            if st.button("🚀 ANALİZİ BAŞLAT VE VERİTABANINA KAYDET"):
                p = st.progress(0)
                scs, decs = [], []
                thr = get_db_data("SELECT value FROM settings WHERE key='risk_threshold'").iloc[0]['value']

                for i, row in df_b.iterrows():
                    # --- DÜZELTME 1: Excel'deki 'Tutar (TL)' başlığına göre veriyi al ---
                    raw_val = row.get('Tutar (TL)') or row.get('Tutar') or row.get('Kredi Tutarı') or 0
                    current_amt = float(str(raw_val).replace(',', ''))

                    # --- DÜZELTME 2: 'Vade' başlığını kontrol et ---
                    current_vade = int(row.get('Vade') or row.get('Vade (Ay)') or 24)

                    inp_b = {
                        'checking_account': maps['checking_account'].get(row.get('Hesap_Durumu'), 'A14'),
                        'duration': current_vade,
                        # --- DÜZELTME 3: 'KKB Geçmişi' başlığını kontrol et ---
                        'credit_history': maps['credit_history'].get(row.get('KKB Geçmişi') or row.get('KKB_Gecmisi'),
                                                                     'A32'),
                        'purpose': maps['purpose'].get(row.get('Amac'), 'A40'),
                        'credit_amount': current_amt / 80,  # DÜZELTME: Buraya da temizlenmiş tutarı koyduk
                        'savings_account': maps['savings_account'].get(row.get('Birikim'), 'A65'),
                        'employment': maps['employment'].get(row.get('Kidem'), 'A73'),
                        'installment_rate': int(row.get('Borclanma_Orani', 2)),
                        'status_sex': 'A93', 'guarantors': 'A101', 'residence_since': 4,
                        'property': maps['property'].get(row.get('Teminat'), 'A121'),
                        'age': int(row.get('Yas') or row.get('Müşteri Yaşı') or 30),
                        'other_installments': 'A143',
                        'housing': maps['housing'].get(row.get('Konut'), 'A152'),
                        'existing_credits': 1,
                        'job': maps['job'].get(row.get('Meslek'), 'A173'),
                        'people_liable': 1, 'telephone': 'A191', 'foreign_worker': 'A201'
                    }

                    try:
                        proc = preprocessor.transform(pd.DataFrame([inp_b]))
                        risk = model.predict(proc, verbose=0)[0][0]
                        f_score, _ = calculate_hybrid_score(int((1 - risk) * 1900), inp_b)

                        k_sonuc = "ONAY" if f_score >= thr else "RED"
                        if current_amt > 750000 and f_score < 1700:
                            k_sonuc = "RED (Yüksek Risk)"

                        k_durum = "MÜDÜR ONAYINDA" if current_amt > 500000 else "TAMAMLANDI"

                        scs.append(f_score)
                        decs.append(k_sonuc)

                        # Veritabanına Kayıt (current_amt artık 0 değil!)
                        m_tc = str(row.get('TC') or row.get('TCKN') or '00000000000')
                        add_history(
                            m_tc,
                            int(inp_b['age']),
                            int(current_amt),
                            current_vade,
                            f_score,
                            k_sonuc,
                            k_durum,
                            st.session_state['name']  # Müdüre kaydet
                        )

                    except Exception as e:
                        scs.append(0)
                        decs.append("HATA")

                    p.progress((i + 1) / len(df_b))

                df_b['AI_Skor'] = scs
                df_b['AI_Karar'] = decs
                st.success(f"✅ {len(df_b)} müşteri başarıyla analiz edildi.")
                st.dataframe(df_b)
                st.rerun()  # Grafiğin hemen güncellenmesi için önemli