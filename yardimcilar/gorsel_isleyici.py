"""
TruthLens AI - Görsel Algılama Modülü
Yazar: TruthLens AI Ekibi
Bu modül, yapay zeka tarafından üretilen görselleri tespit eder
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

class GorselAlgilayici:
    """Gerçek vs AI üretilmiş görselleri tespit eden sınıf"""
    
    def __init__(self):
        self.model = None
        print("✅ Görsel algılayıcı hazır!")
    
    def gorsel_ozellikleri_cikar(self, goruntu):
        """Görselden özellikler çıkarır"""
        # Görseli analiz et
        gri = cv2.cvtColor(goruntu, cv2.COLOR_RGB2GRAY)
        
        # Gürültü seviyesi (AI görselleri genelde daha az gürültülü)
        gurultu = np.std(gri) / 50
        
        # Kenar yoğunluğu
        kenarlar = cv2.Canny(gri, 50, 150)
        kenar_yogunlugu = np.sum(kenarlar > 0) / (gri.shape[0] * gri.shape[1]) * 100
        
        # Renk çeşitliliği
        renk_std = np.std(goruntu, axis=(0, 1)).mean()
        
        # Doku pürüzsüzlüğü (AI görselleri genelde çok pürüzsüz)
        laplacian = cv2.Laplacian(gri, cv2.CV_64F)
        doku_purusuzluk = np.var(laplacian) / 1000
        
        return {
            'gurultu_seviyesi': round(gurultu, 3),
            'kenar_yogunlugu': round(kenar_yogunlugu, 2),
            'renk_cesitliligi': round(renk_std, 3),
            'doku_purusuzluk': round(doku_purusuzluk, 2)
        }
    
    def tahmin_yap(self, image_file):
        """Görselin AI üretilmiş mi yoksa gerçek mi olduğunu tahmin eder"""
        try:
            # Görseli yükle
            image = Image.open(image_file)
            
            # RGB'ye çevir
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # OpenCV formatına çevir
            image_np = np.array(image)
            
            # Görsel boyutlarını al
            genislik, yukseklik = image.size
            
            # Görsel özelliklerini çıkar
            ozellikler = self.gorsel_ozellikleri_cikar(image_np)
            
            # AI skoru hesapla (basit kurallar tabanlı)
            # Bu, gerçek bir model olmadan çalışan akıllı bir algılama sistemi
            
            ai_skor = 0
            
            # Kural 1: Çok pürüzsüz doku -> AI olabilir
            if ozellikler['doku_purusuzluk'] < 50:
                ai_skor += 25
            
            # Kural 2: Düşük gürültü -> AI olabilir  
            if ozellikler['gurultu_seviyesi'] < 15:
                ai_skor += 25
            
            # Kural 3: Çok keskin kenarlar -> AI olabilir
            if ozellikler['kenar_yogunlugu'] > 40:
                ai_skor += 25
            
            # Kural 4: Yüksek renk standardı -> AI olabilir
            if ozellikler['renk_cesitliligi'] > 45:
                ai_skor += 15
            
            # Kural 5: Kare görseller -> AI görsellerde daha yaygın
            if abs(genislik - yukseklik) < 50:
                ai_skor += 10
            
            # Normalizasyon
            ai_skor = min(ai_skor, 99)
            gercek_skor = 100 - ai_skor
            
            # Sonuç belirleme
            if ai_skor > 70:
                etiket = "🤖 YAPAY ZEKA TARAFINDAN ÜRETİLMİŞ"
                yuzde = ai_skor
                diger = "Gerçek Fotoğraf"
                diger_yuzde = gercek_skor
            elif gercek_skor > 70:
                etiket = "✅ GERÇEK FOTOĞRAF"
                yuzde = gercek_skor
                diger = "Yapay Zeka"
                diger_yuzde = ai_skor
            else:
                etiket = "⚠️ KARARSIZ BÖLGE"
                yuzde = max(ai_skor, gercek_skor)
                diger = "Diğer kategori"
                diger_yuzde = min(ai_skor, gercek_skor)
            
            # Öneri oluştur
            oneri = self.oneri_olustur(etiket, yuzde, ozellikler)
            
            return {
                'etiket': etiket,
                'yuzde': round(yuzde, 2),
                'ai_ihtimal': round(ai_skor, 2),
                'gercek_ihtimal': round(gercek_skor, 2),
                'analiz': ozellikler,
                'oneri': oneri,
                'diger_ihtimal': diger,
                'diger_yuzde': round(diger_yuzde, 2),
                'boyutlar': f"{genislik} x {yukseklik}",
                'mod': image.mode
            }
            
        except Exception as e:
            return {
                'etiket': '❌ HATA',
                'yuzde': 0,
                'ai_ihtimal': 0,
                'gercek_ihtimal': 0,
                'analiz': None,
                'oneri': f'⚠️ Görsel yüklenirken hata oluştu: {str(e)}'
            }
    
    def oneri_olustur(self, etiket, yuzde, analiz):
        """Görsel için öneriler sunar"""
        if "YAPAY ZEKA" in etiket and yuzde > 70:
            return """
            💡 **AI Görsel Tespit Edildi:**
            • Görsel çok pürüzsüz ve yapay görünüyor
            • Gerçek fotoğraflarda daha fazla doku ve gürültü olur
            • Kenarlar çok keskin - gerçekte daha yumuşak geçişler olurdu
            • Renk geçişleri çok düzgün
            """
        elif "GERÇEK" in etiket and yuzde > 70:
            return """
            📸 **Bu Görsel Gerçek Görünüyor!**
            • Doğal doku ve gürültü seviyesi
            • Gerçekçi kenar yumuşaklığı
            • Otantik renk dağılımı
            • Profesyonel bir fotoğrafçılık örneği olabilir
            """
        else:
            return """
            🔍 **Kararsız Görsel Analizi:**
            • Bu görsel hem AI hem gerçek özellikleri taşıyor
            • Daha yüksek çözünürlüklü bir versiyonunu deneyin
            • Farklı bir görsel ile tekrar test edin
            """


gorsel_algilayici = GorselAlgilayici()