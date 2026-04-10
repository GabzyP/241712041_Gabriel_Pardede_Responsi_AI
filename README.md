# 😁Responsi AI

Repositori ini berisi Responsi untuk mata kuliah Kecerdasan Buatan. Proyek ini berfokus pada perancangan dan pengoptimalan model *Machine Learning* menggunakan arsitektur *Convolutional Neural Network* (CNN) berbasis **PyTorch** untuk mengklasifikasikan gambar hewan ke dalam 3 kategori: **Cat, Dog, dan Wild**.

Selain tahap *training* model, proyek ini juga mencakup tahap *Deployment* ke dalam bentuk aplikasi web antarmuka sederhana menggunakan **Flask**.

## 📂 Struktur Repositori

- `Responsi_Gabriel_Pardede.ipynb` : *Notebook* utama yang memuat proses ekstraksi data, perancangan arsitektur CNN, serta implementasi teknik mitigasi *overfitting* dan *training loop*.
- `app.py` : Kode *backend* Flask yang berfungsi sebagai *server* web dan jembatan logika untuk memproses gambar yang diunggah ke dalam model AI.
- `best_model.pth` : File memori dari model AI dengan nilai *Validation Loss* terbaik hasil dari proses *training*.
- `templates/index.html` : File *frontend* berbasis HTML yang menjadi antarmuka pengguna untuk melakukan unggah gambar dan melihat hasil prediksi.

## ✨ Fitur Pengoptimalan Model Anti-Overfitting
Model ini telah melalui serangkaian proses optimasi arsitektur untuk memastikan metrik akurasi yang stabil (*Good Fit*), meliputi:
1. **Data Augmentation:** Penggunaan `RandomHorizontalFlip` dan `RandomRotation` untuk memperkaya variasi dataset latih.
2. **Regularization:** Penggunaan `Dropout(0.5)` untuk mencegah ketergantungan berlebih pada neuron tertentu, serta parameter `weight_decay` pada *optimizer*.
3. **Batch Normalization:** Menstabilkan distribusi data di setiap lapisan komputasi untuk mempercepat proses adaptasi model.
4. **Early Stopping & LR Scheduler:** Mekanisme penghentian *training* otomatis pada titik terekstrim untuk mengunci bobot model terbaik sebelum fase *overfitting* terjadi.

## 🚀 Cara Menjalankan Aplikasi Web Lokal

### Prasyarat Sistem
Pastikan Python sudah terinstal di perangkat Anda. Buka Terminal atau Command Prompt, lalu instal *library* yang dibutuhkan dengan perintah berikut:
```bash
pip install flask torch torchvision pillow
python app.py
```
Buka browser dan akses alamat http://127.0.0.1:5000
