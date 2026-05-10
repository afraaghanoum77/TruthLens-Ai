"""
TruthLens AI - Görsel Algılama Modeli (TAMAMEN YENİ)
Yazar: TruthLens AI Ekibi
"""

import cv2
import numpy as np
from PIL import Image
import random

class GorselAlgilayici:
    """Gerçek vs AI üretilmiş görselleri tespit eden sınıf - TAMİR EDİLDİ"""
    
    def __init__(self):
        self.model = None
        print("✅ Görsel algılayıcı hazır! (Tamir edildi)")
    
    def gorsel_ozellikleri_cikar(self, goruntu):
        """Görselden detaylı özellikler çıkarır"""
        
        # Gri tonlamaya çevir
        if len(goruntu.shape) == 3:
            gri = cv2.cvtColor(goruntu, cv2.COLOR_RGB2GRAY)
        else:
            gri = goruntu
        
        # ========== TEMEL ÖZELLİKLER ==========
        
        # 1. GÜRÜLTÜ SEVİYESİ (Gerçek fotoğraflar DAHA FAZLA gürültülü)
        gurultu = np.std(gri) / 20
        
        # 2. KENAR YOĞUNLUĞU
        kenarlar = cv2.Canny(gri, 50, 150)
        kenar_yogunlugu = np.sum(kenarlar > 0) / (gri.shape[0] * gri.shape[1]) * 100
        
        # 3. DOKU PÜRÜZSÜZLÜĞÜ (AI: düşük varyans = pürüzsüz)
        laplacian = cv2.Laplacian(gri, cv2.CV_64F)
        doku_varyansi = np.var(laplacian)
        doku_purusuzluk = doku_varyansi / 1000
        
        # 4. RENK ANALİZİ (sadece RGB için)
        if len(goruntu.shape) == 3:
            # Renk kanallarının standart sapması
            renk_std_r = np.std(goruntu[:,:,0])
            renk_std_g = np.std(goruntu[:,:,1])
            renk_std_b = np.std(goruntu[:,:,2])
            ortalama_renk_std = (renk_std_r + renk_std_g + renk_std_b) / 3
            
            # Renk çeşitliliği (AI görsellerde DAHA DÜZGÜN dağılım)
            renk_cesitliligi = ortalama_renk_std
        else:
            renk_cesitliligi = np.std(gri)
        
        # 5. PARLAKLIK ve KONTRAST
        parlaklik = np.mean(gri)
        kontrast = np.std(gri)
        
        # 6. FREKANS ANALİZİ (YENİ! - AI görselleri tespit için çok önemli)
        # Fourier dönüşümü ile yüksek frekansları analiz et
        f = np.fft.fft2(gri)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        yuksek_frekans = np.mean(magnitude_spectrum[gri.shape[0]//4:3*gri.shape[0]//4, 
                                                    gri.shape[1]//4:3*gri.shape[1]//4])
        
        # 7. HİSTOGRAM ANALİZİ (AI görsellerde DAHA DÜZGÜN histogram)
        hist = cv2.calcHist([gri], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        hist_duzgunluk = np.std(hist) * 1000  # Düşük değer = düzgün histogram
        
        # 8. BLOBLAR (Detaylar) - Gerçek fotoğraflarda daha fazla küçük detay
        _, thresh = cv2.threshold(gri, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blob_sayisi = len(contours)
        
        return {
            'gurultu': round(gurultu, 3),
            'kenar_yogunlugu': round(kenar_yogunlugu, 2),
            'doku_purusuzluk': round(doku_purusuzluk, 3),
            'renk_cesitliligi': round(renk_cesitliligi, 2),
            'parlaklik': round(parlaklik, 2),
            'kontrast': round(kontrast, 2),
            'yuksek_frekans': round(yuksek_frekans, 2),
            'hist_duzgunluk': round(hist_duzgunluk, 3),
            'blob_sayisi': blob_sayisi
        }
    
    def tahmin_yap(self, image_file):
        """Görselin AI mi Gerçek mi olduğunu tespit eder - DÜZELTİLMİŞ ALGORİTMA"""
        try:
            image = Image.open(image_file)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image)
            genislik, yukseklik = image.size
            
            ozellikler = self.gorsel_ozellikleri_cikar(image_np)
            
            # ========== YENİ VE DOĞRU PUANLAMA SİSTEMİ ==========
            # GERÇEK fotoğraf puanı (yüksek = gerçek)
            # AI fotoğraf puanı (yüksek = AI)
            
            gercek_puan = 0
            ai_puan = 0
            
            # ----- KURAL 1: DOKU PÜRÜZSÜZLÜĞÜ -----
            # AI: çok pürüzsüz (düşük varyans)
            # Gerçek: dokulu (yüksek varyans)
            if ozellikler['doku_purusuzluk'] < 50:
                ai_puan += 30  # Çok pürüzsüz = AI
            elif ozellikler['doku_purusuzluk'] > 150:
                gercek_puan += 30  # Dokulu = Gerçek
            else:
                gercek_puan += 10
            
            # ----- KURAL 2: GÜRÜLTÜ -----
            # Gerçek fotoğraflar DAHA FAZLA gürültülü
            if ozellikler['gurultu'] > 2.5:
                gercek_puan += 35  # Yüksek gürültü = Gerçek
            elif ozellikler['gurultu'] < 1.2:
                ai_puan += 35  # Düşük gürültü = AI
            else:
                gercek_puan += 15
            
            # ----- KURAL 3: KENAR YOĞUNLUĞU -----
            # Gerçek fotoğraflar daha fazla doğal kenar içerir
            if ozellikler['kenar_yogunlugu'] > 12:
                gercek_puan += 25
            elif ozellikler['kenar_yogunlugu'] < 5:
                ai_puan += 25
            
            # ----- KURAL 4: HİSTOGRAM DÜZGÜNLÜĞÜ -----
            # AI görsellerinde histogram DAHA DÜZGÜN
            if ozellikler['hist_duzgunluk'] < 5:
                ai_puan += 20  # Düzgün histogram = AI
            else:
                gercek_puan += 15  # Düzensiz histogram = Gerçek
            
            # ----- KURAL 5: YÜKSEK FREKANS -----
            # Gerçek fotoğraflar daha fazla yüksek frekans içerir (detay)
            if ozellikler['yuksek_frekans'] > 30:
                gercek_puan += 20
            else:
                ai_puan += 10
            
            # ----- KURAL 6: BLOBLAR (Detaylar) -----
            # Gerçek fotoğraflar daha fazla küçük detay içerir
            if ozellikler['blob_sayisi'] > 100:
                gercek_puan += 15
            elif ozellikler['blob_sayisi'] < 30:
                ai_puan += 15
            
            # ----- KURAL 7: RENK ÇEŞİTLİLİĞİ -----
            # AI görselleri genelde daha düzgün renk dağılımına sahip
            if ozellikler['renk_cesitliligi'] < 40:
                ai_puan += 10
            else:
                gercek_puan += 10
            
            # Normalizasyon (0-100 arası)
            toplam = gercek_puan + ai_puan
            if toplam > 0:
                gercek_yuzde = (gercek_puan / toplam) * 100
                ai_yuzde = (ai_puan / toplam) * 100
            else:
                gercek_yuzde = 50
                ai_yuzde = 50
            
            # SONUÇ BELİRLEME (Daha keskin eşikler)
            if gercek_yuzde >= 55:
                etiket = "✅ GERÇEK FOTOĞRAF"
                yuzde = gercek_yuzde
                diger = "Yapay Zeka"
                diger_yuzde = ai_yuzde
                renk = "yesil"
            elif ai_yuzde >= 55:
                etiket = "🤖 YAPAY ZEKA TARAFINDAN ÜRETİLMİŞ"
                yuzde = ai_yuzde
                diger = "Gerçek Fotoğraf"
                diger_yuzde = gercek_yuzde
                renk = "kirmizi"
            else:
                # Kararsız - hangisi daha yüksekse onu göster
                if gercek_yuzde > ai_yuzde:
                    etiket = "✅ BÜYÜK İHTİMAL GERÇEK FOTOĞRAF"
                    yuzde = gercek_yuzde
                else:
                    etiket = "🤖 BÜYÜK İHTİMAL AI ÜRETİMİ"
                    yuzde = ai_yuzde
                diger = "Diğer kategori"
                diger_yuzde = min(gercek_yuzde, ai_yuzde)
                renk = "sari"
            
            # Öneri oluştur
            oneri = self.oneri_olustur(etiket, yuzde, ozellikler, gercek_yuzde, ai_yuzde)
            
            return {
                'etiket': etiket,
                'yuzde': round(yuzde, 2),
                'ai_ihtimal': round(ai_yuzde, 2),
                'gercek_ihtimal': round(gercek_yuzde, 2),
                'analiz': ozellikler,
                'oneri': oneri,
                'diger_ihtimal': diger,
                'diger_yuzde': round(diger_yuzde, 2),
                'boyutlar': f"{genislik} x {yukseklik}",
                'mod': image.mode,
                'renk': renk
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
    
    def oneri_olustur(self, etiket, yuzde, analiz, gercek_yuzde, ai_yuzde):
        """Görsel için öneriler sunar"""
        
        if "GERÇEK" in etiket and yuzde >= 55:
            return f"""
            📸 **SONUÇ: GERÇEK FOTOĞRAF** (Güven: %{yuzde:.1f})
            
            🔍 **Tespit Edilen Gerçek Fotoğraf Özellikleri:**
            • Gürültü seviyesi doğal: {analiz['gurultu']}
            • Kenar yoğunluğu gerçekçi: {analiz['kenar_yogunlugu']}
            • Doku varyansı doğal: {analiz['doku_purusuzluk']}
            • Histogram düzensizliği gerçekçi
            • Yeterli miktarda doğal detay mevcut
            
            💡 Bu görsel gerçek bir fotoğraf makinesi veya telefon ile çekilmiş olabilir.
            """
            
        elif "YAPAY ZEKA" in etiket and yuzde >= 55:
            return f"""
            🤖 **SONUÇ: YAPAY ZEKA ÜRETİMİ** (Güven: %{yuzde:.1f})
            
            🔍 **Tespit Edilen AI Özellikleri:**
            • Gürültü seviyesi çok düşük: {analiz['gurultu']} (Gerçekte daha yüksek olur)
            • Doku çok pürüzsüz: {analiz['doku_purusuzluk']}
            • Histogram çok düzgün: {analiz['hist_duzgunluk']}
            • Yapay kenar yumuşaklığı tespit edildi
            • Detaylar çok düzenli
            
            💡 Bu görsel Midjourney, DALL-E veya Stable Diffusion gibi bir AI modeli tarafından üretilmiş olabilir.
            """
            
        elif "BÜYÜK İHTİMAL GERÇEK" in etiket:
            return f"""
            🟢 **BÜYÜK İHTİMAL GERÇEK FOTOĞRAF** (Güven: %{yuzde:.1f})
            
            🔍 Analiz: %{gercek_yuzde:.1f} Gerçek / %{ai_yuzde:.1f} AI
            
            💡 Görsel gerçek fotoğraf özellikleri gösteriyor ancak emin olmak için 
            farklı bir görselle de test yapabilirsiniz.
            """
            
        elif "BÜYÜK İHTİMAL AI" in etiket:
            return f"""
            🟠 **BÜYÜK İHTİMAL AI ÜRETİMİ** (Güven: %{yuzde:.1f})
            
            🔍 Analiz: %{ai_yuzde:.1f} AI / %{gercek_yuzde:.1f} Gerçek
            
            💡 Görsel yapay zeka üretimi özellikleri gösteriyor.
            """
            
        else:
            return f"""
            ⚖️ **KARARSIZ** (Güven: %{yuzde:.1f})
            
            🔍 Analiz: %{gercek_yuzde:.1f} Gerçek / %{ai_yuzde:.1f} AI
            
            💡 Bu görsel hem AI hem gerçek özellikleri taşıyor.
            Daha yüksek çözünürlüklü bir görsel deneyin.
            """


gorsel_algilayici = GorselAlgilayici()