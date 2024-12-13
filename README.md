# PBL Perceptron Pengenalan Pola Huruf Alphabet

Proyek ini menggunakan algoritma **Perceptron** untuk mengenali pola huruf alfabet yang digambar oleh pengguna pada grid 5x5. Proyek ini dibangun dengan menggunakan HTML untuk tampilan antarmuka pengguna (frontend), dan Flask untuk mengelola backend serta menangani endpoint pengenalan pola.

## Fitur

- **Grid 5x5**: Pengguna dapat menggambar pola huruf alfabet dengan mengklik kotak-kotak pada grid 5x5.
- **Pengenalan Pola**: Setelah pola digambar, sistem akan mengidentifikasi huruf yang dimasukkan menggunakan algoritma Perceptron.
- **Backend Flask**: Flask berfungsi untuk menangani permintaan pengenalan pola dan memberikan hasil prediksi.
- **Interaktivitas**: Pengguna dapat menggambar ulang pola dan mencoba pengenalan untuk huruf yang berbeda.

## Teknologi yang Digunakan

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: Algoritma Perceptron
- **Library**: Flask, NumPy (untuk komputasi), Matplotlib (opsional untuk visualisasi)


## Model Perceptron
Model Perceptron diimplementasikan dalam file model/perceptron.py. Model ini telah dilatih dengan data pola huruf 5x5 untuk mengenali alfabet A-Z. Proses pelatihan dapat dilakukan ulang jika diperlukan, dan kode pelatihan disertakan dalam file ini.

## Pelatihan dan Pengujian  
- Pelatihan: Coba masukkan beberapa pola huruf dan periksa apakah sistem mengenali huruf dengan benar.
- Pengujian Model: Model dapat diuji dengan dataset yang lebih besar untuk meningkatkan akurasi pengenalan.
