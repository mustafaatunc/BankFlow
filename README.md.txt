# 🛠️ Kurulum (Installation)

Projeyi kendi bilgisayarınızda çalıştırmak için adımları izleyin:

1. **Repoyu indirin:**
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/REPO_ADIN.git](https://github.com/KULLANICI_ADIN/REPO_ADIN.git)
   cd REPO_ADIN
   ```

2. **Gerekli kütüphaneleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Admin Şifresini Belirleyin:**
   Ana dizinde `.env` adında bir dosya oluşturun ve içine admin girişi için kullanmak istediğiniz şifreyi yazın:
   ```text
   ADMIN_PASSWORD=BurayaIstediginSifreyiYaz
   ```
   *(Eğer bu dosyayı oluşturmazsanız uygulama güvenlik gereği çalışmayacaktır.)*

4. **Uygulamayı Başlatın:**
   ```bash
   python -m streamlit run app.py
   ```

5. **Giriş Bilgileri:**
   Uygulama açıldığında şu bilgilerle giriş yapabilirsiniz:
   * **E-posta:** admin@admin.com
   * **Şifre:** `.env` dosyasına yazdığınız şifre.