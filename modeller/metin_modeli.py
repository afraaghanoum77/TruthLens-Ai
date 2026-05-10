"""
TruthLens AI - Metin Algılama Modeli
Yazar: TruthLens AI Ekibi
"""

import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# NLTK kaynaklarını indir (ilk çalıştırmada)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class MetinAlgilayici:
    """Yapay Zeka vs İnsan metinlerini tespit eden sınıf"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_olustur()
    
    def metin_temizle(self, metin):
        """Metni temizler ve normalize eder"""
        metin = metin.lower()
        metin = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ]', '', metin)
        metin = re.sub(r'\d+', '', metin)
        metin = re.sub(r'\s+', ' ', metin).strip()
        return metin
    
    def metni_analiz_et(self, metin):
        """Metni kelimelere ayır ve analiz et"""
        kelime_sayisi = len(metin.split())
        cumle_sayisi = len(re.findall(r'[.!?]+', metin))
        ozel_karakterler = len(re.findall(r'[!@#$%^&*()]', metin))
        
        kelimeler = metin.split()
        ortalama_uzunluk = np.mean([len(k) for k in kelimeler]) if kelimeler else 0
        
        return {
            'kelime_sayisi': kelime_sayisi,
            'cumle_sayisi': cumle_sayisi,
            'ortalama_kelime_uzunlugu': round(ortalama_uzunluk, 2),
            'ozel_karakter_sayisi': ozel_karakterler
        }
    
    def model_olustur(self):
        """TF-IDF + Logistic Regression modeli oluştur"""
        # Örnek eğitim verileri (hazır)
        metinler = [
            # AI tarafından üretilmiş metinler (0)
            "Yapay zeka teknolojileri günümüzde hızla gelişmektedir. Makine öğrenmesi algoritmaları birçok alanda kullanılmaktadır.",
            "Veri bilimi ve büyük veri analitiği, modern iş dünyasının temel taşlarıdır. Şirketler veri odaklı kararlar almaktadır.",
            "Derin öğrenme sinir ağları, görüntü tanıma ve doğal dil işleme gibi alanlarda çığır açmıştır.",
            "Bulut bilişim sistemleri, ölçeklenebilirlik ve esneklik sağlayarak işletmelerin dijital dönüşümünü hızlandırmaktadır.",
            "Nesnelerin interneti cihazları, günlük hayatımızda giderek daha fazla yer kaplamaktadır.",
            "Yapay zeka sistemleri, karmaşık problemleri çözmek için tasarlanmış algoritmalar bütünüdür.",
            "Makine öğrenmesi modelleri, veri setlerindeki desenleri tanımak için eğitilir.",
            
            # İnsan tarafından yazılmış metinler (1)
            "Dün arkadaşımla sinemaya gittik. Film çok güzeldi ama biraz uzundu. Çıkışta kahve içtik ve sohbet ettik.",
            "Annemin yaptığı yemekler dünyanın en güzeli. Özellikle böreği nefis olur. Her pazar ailecek toplanırız.",
            "Okula giderken yolda bir kedi yavrusu gördüm. Çok tatlıydı. Onu sevdim ve biraz süt verdim.",
            "Bu yaz tatile Antalya'ya gideceğiz. Deniz kenarında güzel bir otel bulduk. Heyecanla bekliyoruz.",
            "Kitap okumayı çok seviyorum. Özellikle polisiye romanlar favorim. Şu anda üç kitap birden okuyorum.",
            "Dün akşam eski bir arkadaşımı gördüm. 10 yıldır görmemiştim. Çok değişmiş ama yine aynı sıcak insandı.",
            "Balkonda çiçeklerimi sularken kuşların cıvıltısını dinliyorum. Çok huzurlu bir sabah.",
            "Kahvaltıda menemen yaptım. İçine biraz kaşar ekledim çok lezzetli oldu. Afiyetle yedik."
        ]
        
        etiketler = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # 0:AI, 1:İnsan
        
        # Metinleri temizle
        temiz_metinler = [self.metin_temizle(m) for m in metinler]
        
        # TF-IDF vektörleştirici
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(temiz_metinler)
        y = np.array(etiketler)
        
        # Model eğitimi
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X, y)
        
        print("✅ Metin modeli başarıyla oluşturuldu!")
    
    def tahmin_yap(self, metin):
        """Girilen metnin AI veya İnsan olduğunu tahmin eder"""
        if not metin or len(metin.strip()) < 10:
            return {
                'etiket': '⚠️ Yetersiz Metin',
                'yuzde': 0,
                'ai_ihtimal': 0,
                'insan_ihtimal': 0,
                'analiz': None,
                'oneri': '🔴 **Lütfen daha uzun bir metin girin!** (En az 10 karakter)'
            }
        
        # Metin analizi
        metin_analiz = self.metni_analiz_et(metin)
        
        # Metni temizle
        temiz_metin = self.metin_temizle(metin)
        X = self.vectorizer.transform([temiz_metin])
        
        # Olasılıkları hesapla
        olasiliklar = self.model.predict_proba(X)[0]
        ai_olasilik = olasiliklar[0] * 100
        insan_olasilik = olasiliklar[1] * 100
        
        # Tahmin
        tahmin = self.model.predict(X)[0]
        
        if tahmin == 1:
            etiket = "✅ İNSAN TARAFINDAN YAZILMIŞ"
            yuzde = insan_olasilik
            diger = "Yapay Zeka"
            diger_yuzde = ai_olasilik
        else:
            etiket = "🤖 YAPAY ZEKA TARAFINDAN ÜRETİLMİŞ"
            yuzde = ai_olasilik
            diger = "İnsan"
            diger_yuzde = insan_olasilik
        
        # Öneri oluştur
        oneri = self.oneri_olustur(etiket, yuzde, metin_analiz)
        
        return {
            'etiket': etiket,
            'yuzde': round(yuzde, 2),
            'ai_ihtimal': round(ai_olasilik, 2),
            'insan_ihtimal': round(insan_olasilik, 2),
            'analiz': metin_analiz,
            'oneri': oneri,
            'diger_ihtimal': diger,
            'diger_yuzde': round(diger_yuzde, 2)
        }
    
    def oneri_olustur(self, etiket, yuzde, analiz):
        """Kullanıcıya öneriler sunar"""
        if "YAPAY ZEKA" in etiket and yuzde > 70:
            return """
            💡 **Öneriler:**
            • Bu metin büyük olasılıkla yapay zeka tarafından üretilmiş
            • Daha kişisel ve duygusal ifadeler eklemeyi deneyin
            • Kendi deneyimlerinizden örnekler verin
            • Samimi bir dil kullanmaya çalışın
            • Kısa cümleler ve günlük konuşma dili kullanın
            """
        elif "İNSAN" in etiket and yuzde > 70:
            return """
            🎉 **Harika! Metniniz doğal görünüyor!**
            • Kişisel anlatımınız çok başarılı
            • Duygusal ifadeler metninizi güçlendirmiş
            • Bu şekilde devam edin! ✨
            • Metniniz samimi ve içten görünüyor
            """
        elif yuzde < 60:
            return """
            ⚖️ **Kararsız Bölge:**
            • Metniniz hem AI hem de insan özellikleri gösteriyor
            • Daha fazla kişisel yorum eklemeyi deneyin
            • Ya da metni biraz daha uzatın
            • Duygularınızı daha fazla ifade edin
            """
        else:
            return """
            📝 **Genel Öneriler:**
            • Metninize kendi düşüncelerinizi ekleyin
            • Kişisel anılarınızdan bahsedin
            • Duygusal ifadeler kullanın
            • Kısa ve net cümleler kurun
            """


# Tek örnek oluştur
metin_algilayici = MetinAlgilayici()