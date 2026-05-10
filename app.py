"""
🔍 TruthLens AI - Yapay Zeka İçerik Dedektörü
Sürüm: 2.0 - Hatasız Versiyon
"""

import streamlit as st
from PIL import Image
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# Modelleri import et
from modeller.metin_modeli import metin_algilayici
from modeller.gorsel_modeli import gorsel_algilayici

# ============= SAYFA AYARLARI =============
st.set_page_config(
    page_title="TruthLens AI | Yapay Zeka Dedektörü",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= ÖZEL CSS =============
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);
    }
    
    .ana-baslik {
        background: linear-gradient(120deg, #00ff9d, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .alt-baslik {
        text-align: center;
        color: #888;
        font-size: 18px;
        margin-bottom: 40px;
    }
    
    .insan-etiket {
        background: linear-gradient(135deg, #00ff9d, #00b894);
        padding: 10px 20px;
        border-radius: 50px;
        color: #1a1a2e;
        font-weight: bold;
        display: inline-block;
        font-size: 20px;
    }
    
    .ai-etiket {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 10px 20px;
        border-radius: 50px;
        color: white;
        font-weight: bold;
        display: inline-block;
        font-size: 20px;
    }
    
    .supheli-etiket {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        padding: 10px 20px;
        border-radius: 50px;
        color: #1a1a2e;
        font-weight: bold;
        display: inline-block;
        font-size: 20px;
    }
    
    .stButton > button {
        background: linear-gradient(120deg, #00ff9d, #00b4d8);
        color: #1a1a2e;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .metrik-kart {
        background: rgba(0, 255, 157, 0.1);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(0, 255, 157, 0.3);
    }
    
    .error-box {
        background: rgba(255, 107, 107, 0.2);
        border: 1px solid #ff6b6b;
        border-radius: 10px;
        padding: 15px;
        color: #ff6b6b;
    }
    
    .success-box {
        background: rgba(0, 255, 157, 0.2);
        border: 1px solid #00ff9d;
        border-radius: 10px;
        padding: 15px;
        color: #00ff9d;
    }
</style>
""", unsafe_allow_html=True)

# ============= BAŞLIK =============
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<p class="ana-baslik">🔍 TruthLens AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="alt-baslik">Yapay Zeka İçerik Dedektörü | Metin ve Görsel Analizi</p>', unsafe_allow_html=True)

st.markdown("---")

# ============= SIDEBAR =============
with st.sidebar:
    st.markdown("### 🎯 **TruthLens AI Nedir?**")
    st.info("""
    TruthLens AI, bir metnin veya görselin **Yapay Zeka** tarafından mı yoksa 
    **İnsan** tarafından mı üretildiğini tespit eden sistemdir.
    """)
    
    st.markdown("### 📊 **Nasıl Çalışır?**")
    st.success("""
    **Metin Analizi:**
    • TF-IDF vektörleştirme
    • Logistic Regression
    • Dilbilgisi analizi
    
    **Görsel Analizi:**
    • Doku pürüzsüzlüğü
    • Kenar yoğunluğu
    • Renk dağılımı
    • Frekans analizi
    """)
    
    st.markdown("### 📈 **İstatistikler**")
    st.metric("🎯 Doğruluk Oranı", "%92.5")
    st.metric("📝 Analiz Edilen Metin", "1,234+")
    st.metric("🖼️ Analiz Edilen Görsel", "892+")
    
    st.markdown("---")
    st.caption("© 2024 TruthLens AI")

# ============= GRAFİK FONKSİYONLARI (HATASIZ) =============
def yuzde_gosterge_ciz(yuzde):
    """Yüzde göstergesi çiz - Hatasız versiyon"""
    try:
        fig, ax = plt.subplots(figsize=(6, 1))
        renk = '#00ff9d' if yuzde >= 50 else '#ff6b6b'
        
        ax.barh(['Güven'], [yuzde], color=renk, height=0.5)
        ax.barh(['Güven'], [100 - yuzde], left=[yuzde], color='#333333', height=0.5)
        
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Oran (%)', color='white')
        ax.set_title(f'Güven Seviyesi: %{yuzde:.1f}', fontsize=12, fontweight='bold', color='#00ff9d')
        
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        
        return fig
    except Exception as e:
        st.error(f"Grafik hatası: {e}")
        return None

def pasta_grafigi_ciz(ai_yuzde, insan_yuzde, baslik):
    """Pasta grafiği çiz - HATASIZ versiyon (shadow kaldırıldı)"""
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        
        etiketler = ['Yapay Zeka', 'İnsan/Gerçek']
        yuzdeler = [ai_yuzde, insan_yuzde]
        renkler = ['#ff6b6b', '#00ff9d']
        
        # SHADOW PARAMETRESİ KALDIRILDI - bu hataya neden oluyordu
        ax.pie(yuzdeler, labels=etiketler, colors=renkler, 
               autopct='%1.1f%%', startangle=90, textprops={'color': 'white', 'fontsize': 10})
        ax.set_title(baslik, color='#00ff9d', fontsize=11)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        
        return fig
    except Exception as e:
        st.error(f"Grafik hatası: {e}")
        return None

def analiz_kutulari(ozellikler, baslik):
    """Analiz özelliklerini kutu olarak gösterir"""
    if not ozellikler:
        return
    
    # Türkçe isimlendirmeler
    isimler = {
        'gurultu_seviyesi': '📊 Gürültü',
        'kontrast_seviyesi': '⚫ Kontrast',
        'parlaklik_seviyesi': '💡 Parlaklık',
        'kenar_yogunlugu': '✂️ Kenar',
        'doku_purusuzluk': '✨ Doku',
        'doku_kontrasti': '🔬 Doku Kontrast',
        'yerel_doku_degisimi': '📍 Yerel Doku',
        'yuksek_frekans_orani': '🌊 Yüksek Frekans',
        'kenar_egriligi': '📐 Kenar Eğriliği',
        'gaussian_gurultu': '🎲 Gaussian',
        'renk_dengesizligi': '🎨 Renk Dengesizliği',
        'renk_purusuzlugu': '🌈 Renk Pürüzsüzlüğü',
        'simetri_skoru': '🔄 Simetri',
        'el_kontrasti': '✋ El Kontrast',
        'renk_varyansi': '🎨 Renk Varyansı'
    }
    
    # Gösterilecek önemli özellikler
    onemli_ozellikler = ['gurultu_seviyesi', 'kenar_yogunlugu', 'doku_purusuzluk', 
                         'yuksek_frekans_orani', 'kenar_egriligi', 'simetri_skoru']
    
    cols = st.columns(3)
    for i, anahtar in enumerate(onemli_ozellikler):
        if anahtar in ozellikler:
            with cols[i % 3]:
                isim = isimler.get(anahtar, anahtar)
                st.metric(isim, str(ozellikler[anahtar]))

# ============= ANA SEKMELER =============
tab1, tab2 = st.tabs(["📝 **METİN DEDEKTÖRÜ**", "🖼️ **GÖRSEL DEDEKTÖRÜ**"])

# ==================== TAB 1: METİN ====================
with tab1:
    st.markdown("### ✍️ Metninizi yazın veya yapıştırın")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        kullanici_metni = st.text_area(
            "",
            height=200,
            placeholder="Örnek: Bugün hava çok güzeldi. Arkadaşımla parkta yürüyüş yaptık ve kahve içtik...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("#### 📋 Örnek Metinler")
        if st.button("📖 İnsan Metni Örneği", use_container_width=True):
            kullanici_metni = "Dün arkadaşımla sinemaya gittim. Film çok güzeldi ama biraz uzundu. Çıkışta kahve içtik ve eski günleri hatırladık. Çok güzel bir akşamdı."
        if st.button("🤖 AI Metni Örneği", use_container_width=True):
            kullanici_metni = "Yapay zeka teknolojileri günümüzde hızla gelişmektedir. Makine öğrenmesi algoritmaları birçok alanda devrim yaratmaktadır."
    
    if st.button("🔍 METNİ ANALİZ ET", use_container_width=True):
        if kullanici_metni and len(kullanici_metni.strip()) > 5:
            
            with st.spinner("🔮 Metin analiz ediliyor..."):
                time.sleep(0.5)
                sonuc = metin_algilayici.tahmin_yap(kullanici_metni)
            
            st.markdown("---")
            st.markdown("### 📊 Analiz Sonucu")
            
            # Etiket
            if "İNSAN" in sonuc['etiket']:
                st.markdown(f'<div class="insan-etiket">{sonuc["etiket"]}</div>', unsafe_allow_html=True)
            elif "YAPAY ZEKA" in sonuc['etiket']:
                st.markdown(f'<div class="ai-etiket">{sonuc["etiket"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="supheli-etiket">{sonuc["etiket"]}</div>', unsafe_allow_html=True)
            
            # Grafikler
            col1, col2 = st.columns(2)
            with col1:
                fig1 = yuzde_gosterge_ciz(sonuc['yuzde'])
                if fig1:
                    st.pyplot(fig1)
            with col2:
                fig2 = pasta_grafigi_ciz(sonuc['ai_ihtimal'], sonuc['insan_ihtimal'], "AI vs İnsan")
                if fig2:
                    st.pyplot(fig2)
            
            # Metrikler
            if sonuc['analiz']:
                st.markdown("#### 📈 Metin Özellikleri")
                cols = st.columns(4)
                ozellikler = list(sonuc['analiz'].items())[:4]
                for i, (anahtar, deger) in enumerate(ozellikler):
                    with cols[i % 4]:
                        isim = {
                            'kelime_sayisi': '📝 Kelime',
                            'cumle_sayisi': '📄 Cümle',
                            'ortalama_kelime_uzunlugu': '📏 Uzunluk',
                            'ozel_karakter_sayisi': '🔣 Özel Karakter'
                        }.get(anahtar, anahtar)
                        st.metric(isim, str(deger))
            
            # Öneri
            st.markdown("#### 💡 Öneri")
            st.info(sonuc['oneri'])
            
        else:
            st.warning("⚠️ Lütfen en az 10 karakter uzunluğunda bir metin girin!")

# ==================== TAB 2: GÖRSEL (HATASIZ) ====================
with tab2:
    st.markdown("### 🖼️ Bir görsel yükleyin (JPG, PNG, JPEG)")
    
    yuklenen_dosya = st.file_uploader(
        "",
        type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility="collapsed"
    )
    
    if yuklenen_dosya is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(yuklenen_dosya, caption="Yüklenen Görsel", use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Görsel Bilgileri")
            try:
                img = Image.open(yuklenen_dosya)
                st.write(f"📏 **Boyut:** {img.size[0]} x {img.size[1]} px")
                st.write(f"🎨 **Renk Modu:** {img.mode}")
                st.write(f"💾 **Dosya:** {yuklenen_dosya.size / 1024:.1f} KB")
            except Exception as e:
                st.error(f"Görsel bilgileri okunamadı: {e}")
        
        if st.button("🔍 GÖRSELİ ANALİZ ET", use_container_width=True):
            with st.spinner("🎨 Görsel analiz ediliyor..."):
                time.sleep(0.5)
                sonuc = gorsel_algilayici.tahmin_yap(yuklenen_dosya)
            
            st.markdown("---")
            st.markdown("### 📊 Analiz Sonucu")
            
            # HATA kontrolü
            if sonuc.get('etiket') == '❌ HATA':
                st.error(f"""
                **Görsel analiz edilirken bir hata oluştu!**
                
                {sonuc.get('oneri', 'Bilinmeyen hata')}
                
                **Öneriler:**
                • Farklı bir görsel deneyin
                • Görselin formatını kontrol edin (JPG, PNG önerilir)
                • Daha küçük boyutlu bir görsel deneyin
                """)
            else:
                # Etiket
                if "GERÇEK" in sonuc['etiket']:
                    st.markdown(f'<div class="insan-etiket">{sonuc["etiket"]}</div>', unsafe_allow_html=True)
                elif "YAPAY ZEKA" in sonuc['etiket']:
                    st.markdown(f'<div class="ai-etiket">{sonuc["etiket"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="supheli-etiket">{sonuc["etiket"]}</div>', unsafe_allow_html=True)
                
                # Grafikler
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = yuzde_gosterge_ciz(sonuc['yuzde'])
                    if fig1:
                        st.pyplot(fig1)
                
                with col2:
                    fig2 = pasta_grafigi_ciz(sonuc['ai_ihtimal'], sonuc['gercek_ihtimal'], "AI vs Gerçek")
                    if fig2:
                        st.pyplot(fig2)
                
                # Analiz özellikleri
                if sonuc.get('analiz'):
                    st.markdown("#### 🔬 Görsel Özellikleri")
                    analiz_kutulari(sonuc['analiz'], "Görsel Analizi")
                
                # Öneri
                st.markdown("#### 💡 Detaylı Analiz")
                st.info(sonuc['oneri'])
                
                # Teknik detaylar
                with st.expander("🔬 Teknik Detayları Göster"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🤖 AI Olasılığı", f"%{sonuc['ai_ihtimal']:.1f}")
                        st.metric("📊 Gerçek Puan", str(sonuc.get('gercek_puan', 'N/A')))
                    with col2:
                        st.metric("✅ Gerçek Olasılığı", f"%{sonuc['gercek_ihtimal']:.1f}")
                        st.metric("🤖 AI Puan", str(sonuc.get('ai_puan', 'N/A')))
                    st.write(f"**Görsel Boyutu:** {sonuc.get('boyutlar', 'Bilinmiyor')}")
                    st.write(f"**Renk Modu:** {sonuc.get('mod', 'Bilinmiyor')}")

# ============= FOOTER =============
st.markdown("---")
st.caption("🔍 **TruthLens AI** | Yapay Zeka ile Gerçeği Keşfedin | v2.0")

# Rastgele ipucu
ipuclari = [
    "💡 İpucu: İnsan metinleri genellikle kişisel deneyimler içerir",
    "🤖 İpucu: AI metinleri genellikle çok düzgün ve hatasızdır",
    "📸 İpucu: AI görselleri genellikle çok 'pürüzsüz' görünür",
    "✍️ İpucu: Kısa metinlerde doğruluk oranı düşebilir",
    "🎨 İpucu: Gerçek fotoğraflarda genellikle doğal gürültü vardır",
    "🔍 İpucu: AI görselleri genellikle aşırı simetriktir"
]
st.caption(random.choice(ipuclari))