# Laporan Proyek Machine Learning - Muhammad Mahathir

## Domain Proyek

Penyakit jantung merupakan salah satu penyebab kematian tertinggi di dunia, dengan sekitar 17,9 juta kematian setiap tahunnya menurut World Health Organization (WHO) [1]. Di Indonesia sendiri, prevalensi penyakit jantung berdasarkan diagnosis dokter sebesar 1,5% dengan provinsi tertinggi di Kalimantan Utara (2,2%) menurut data Riskesdas 2018 [2]. Kondisi ini menjadi tantangan serius dalam bidang kesehatan karena penyakit jantung seringkali tidak menunjukkan gejala yang jelas pada tahap awal.

Tantangan utama dalam penanganan penyakit jantung adalah deteksi dini yang sulit dilakukan tanpa peralatan medis canggih. Banyak fasilitas kesehatan, terutama di daerah dengan sumber daya terbatas, tidak memiliki akses ke peralatan diagnostik canggih seperti angiografi koroner atau ekokardiografi. Akibatnya, banyak kasus penyakit jantung terdiagnosis pada tahap lanjut ketika pengobatan menjadi lebih sulit dan hasil yang kurang optimal [3].

Machine learning dapat menjadi solusi yang menjanjikan untuk masalah ini, dengan memanfaatkan data pasien yang tersedia seperti usia, jenis kelamin, tekanan darah, hasil tes darah, dan gejala yang dilaporkan untuk memprediksi risiko penyakit jantung. Pendekatan ini memungkinkan skrining awal yang efisien, hemat biaya, dan dapat diakses secara luas, membantu tenaga medis mengidentifikasi pasien yang membutuhkan pemeriksaan lebih lanjut.

Penelitian sebelumnya telah menunjukkan bahwa algoritma machine learning dapat mencapai akurasi tinggi dalam memprediksi penyakit jantung. Sebagai contoh, Shah et al. (2020) menggunakan algoritma Random Forest dan mencapai akurasi 88,7% dalam memprediksi penyakit jantung menggunakan dataset Cleveland Heart Disease [4], sementara Mohan et al. (2019) mengembangkan model hybrid menggunakan Random Forest dan teknik optimasi untuk mencapai akurasi 88,4% [5].

Proyek ini bertujuan untuk mengembangkan model klasifikasi yang dapat diandalkan untuk memprediksi risiko penyakit jantung menggunakan dataset Heart Failure Prediction, dengan fokus pada perbandingan berbagai algoritma machine learning untuk menemukan pendekatan optimal.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut adalah pernyataan masalah dalam proyek ini:

1. Deteksi dini penyakit jantung sulit dilakukan tanpa alat diagnostik canggih, terutama di fasilitas kesehatan dengan sumber daya terbatas, mengakibatkan keterlambatan diagnosis dan hasil pengobatan yang kurang optimal.
2. Data medis pasien yang kompleks memerlukan analisis prediktif yang akurat untuk mengidentifikasi pasien berisiko tinggi penyakit jantung, namun perlu pendekatan yang tepat untuk mendapatkan hasil prediksi yang dapat diandalkan.
3. Ketidakseimbangan dalam distribusi kelas pada data medis dapat menyebabkan bias dalam model prediksi, mempengaruhi kemampuan sistem untuk mengidentifikasi kasus positif dengan benar.

### Goals

Tujuan dari proyek ini adalah:

1. Mengembangkan model machine learning yang dapat memprediksi keberadaan penyakit jantung pada pasien dengan akurasi, precision, dan recall yang tinggi, sehingga dapat digunakan sebagai alat bantu diagnostik awal.
2. Membandingkan performa beberapa algoritma machine learning (Logistic Regression, Random Forest, dan Gradient Boosting) untuk menemukan pendekatan optimal dalam memprediksi penyakit jantung.
3. Mengoptimalkan model terpilih melalui hyperparameter tuning untuk meningkatkan performa prediksi.

### Solution Statements

Untuk mencapai tujuan tersebut, berikut adalah solusi yang ditawarkan:

1. Mengimplementasikan tiga algoritma machine learning untuk perbandingan:
   - Logistic Regression: Sebagai model dasar yang interpretable dan efektif untuk klasifikasi biner.
   - Random Forest Classifier: Model ensemble yang tahan terhadap overfitting dan dapat menangani hubungan non-linear antar fitur.
   - Gradient Boosting Classifier: Model ensemble kuat yang dapat menangkap hubungan kompleks dalam data.

2. Menerapkan teknik SMOTE (Synthetic Minority Oversampling Technique) untuk mengatasi ketidakseimbangan kelas dalam dataset, sehingga meningkatkan performa model dalam mengidentifikasi kasus positif.

3. Melakukan hyperparameter tuning pada algoritma terbaik menggunakan RandomizedSearchCV untuk meningkatkan performa model, dengan fokus pada F1-score sebagai metrik utama untuk menyeimbangkan precision dan recall.

4. Mengevaluasi model akhir menggunakan metrik yang komprehensif: accuracy, precision, recall, F1-score, dan ROC-AUC, untuk memastikan model memiliki performa yang baik dalam berbagai aspek.

## Data Understanding

Pada proyek ini, dataset yang digunakan adalah Heart Failure Prediction Dataset dari Kaggle. Dataset ini berisi 918 sampel data pasien dengan 11 fitur medis serta satu kolom target yang menunjukkan keberadaan penyakit jantung.

**Sumber Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

Dataset ini dibuat dengan menggabungkan 5 dataset penyakit jantung yang dikenal, dengan duplikasi yang dihilangkan. Dataset ini diproses agar variabel yang digunakan konsisten di semua dataset gabungan.

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:

1. **Age**: Usia pasien dalam tahun (numerik)
2. **Sex**: Jenis kelamin pasien (M = Pria, F = Wanita)
3. **ChestPainType**: Tipe nyeri dada pasien (kategori)
   - TA: Typical Angina
   - ATA: Atypical Angina
   - NAP: Non-Anginal Pain
   - ASY: Asymptomatic
4. **RestingBP**: Tekanan darah istirahat dalam mm Hg (numerik)
5. **Cholesterol**: Kadar kolesterol serum dalam mg/dL (numerik)
6. **FastingBS**: Gula darah puasa (kategori)
   - 0: FastingBS ≤ 120 mg/dL
   - 1: FastingBS > 120 mg/dL
7. **RestingECG**: Hasil elektrokardiogram istirahat (kategori)
   - Normal: Normal
   - ST: Memiliki abnormalitas gelombang ST-T
   - LVH: Menunjukkan kemungkinan atau jelas hipertrofi ventrikel kiri
8. **MaxHR**: Denyut jantung maksimum yang dicapai (numerik)
9. **ExerciseAngina**: Angina yang dipicu oleh olahraga (Y = Ya, N = Tidak)
10. **Oldpeak**: Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat (numerik)
11. **ST_Slope**: Kemiringan segmen ST pada puncak olahraga (kategori)
    - Up: Upsloping
    - Flat: Flat
    - Down: Downsloping
12. **HeartDisease**: Target, menunjukkan keberadaan penyakit jantung (0 = Tidak, 1 = Ya)

### Eksplorasi Data Awal

Untuk memahami dataset dengan lebih baik, beberapa analisis eksplorasi data telah dilakukan:

#### Statistik Deskriptif

Dari hasil analisis deskriptif, terlihat bahwa:
- Rentang usia pasien adalah 28-77 tahun dengan rata-rata 53,5 tahun.
- Tekanan darah istirahat (RestingBP) memiliki range dari 0 hingga 200 mm Hg, dengan nilai 0 yang merupakan anomali.
- Kolesterol serum (Cholesterol) memiliki range dari 0 hingga 603 mg/dL, dengan nilai 0 yang juga merupakan anomali.
- Denyut jantung maksimum (MaxHR) berkisar antara 60-202 bpm.

#### Pengecekan Data

Pengecekan awal menunjukkan bahwa:
- Tidak terdapat missing value dalam dataset.
- Tidak terdapat data duplikat.
- Terdapat beberapa nilai tidak valid:
  - 1 data dengan RestingBP <= 0
  - 172 data dengan Cholesterol = 0

#### Distribusi Kelas Target

Analisis distribusi kelas target (HeartDisease) menunjukkan bahwa:
- 508 sampel (55,3%) merupakan kasus positif (memiliki penyakit jantung)
- 410 sampel (44,7%) merupakan kasus negatif (tidak memiliki penyakit jantung)

Meskipun terdapat sedikit ketidakseimbangan, distribusi kelas masih cukup baik. Namun, untuk meningkatkan performa model, teknik SMOTE akan tetap diterapkan pada data latih.

#### Hubungan Antar Variabel

Dari visualisasi korelasi antar variabel numerik, dapat dilihat bahwa:
- Terdapat korelasi negatif moderat antara Age dan MaxHR (-0,38), yang secara klinis masuk akal karena denyut jantung maksimum cenderung menurun seiring bertambahnya usia.
- Oldpeak menunjukkan korelasi positif dengan HeartDisease (0,4), mengindikasikan bahwa semakin tinggi depresi ST, semakin tinggi kemungkinan seseorang memiliki penyakit jantung.
- MaxHR memiliki korelasi negatif dengan HeartDisease (-0,4), menunjukkan bahwa pasien dengan penyakit jantung cenderung memiliki denyut jantung maksimum yang lebih rendah.

Untuk fitur kategorikal:
- ChestPainType ASY (asymptomatic) menunjukkan asosiasi kuat dengan penyakit jantung.
- ExerciseAngina 'Y' berhubungan erat dengan keberadaan penyakit jantung.
- ST_Slope 'Flat' dan 'Down' lebih sering terjadi pada pasien dengan penyakit jantung.

## Data Preparation

Data preparation dilakukan untuk memastikan dataset siap digunakan dalam pemodelan. Berikut adalah tahapan-tahapan yang dilakukan:

### 1. Penanganan Nilai Tidak Valid

```python
# Menghapus baris dengan RestingBP <= 0
df_clean = df_clean[df_clean['RestingBP'] > 0]

# Mengganti Cholesterol == 0 dengan median
cholesterol_median = df_clean[df_clean['Cholesterol'] > 0]['Cholesterol'].median()
df_clean.loc[df_clean['Cholesterol'] == 0, 'Cholesterol'] = cholesterol_median

# Mengganti Oldpeak negatif dengan 0
df_clean.loc[df_clean['Oldpeak'] < 0, 'Oldpeak'] = 0
```

**Alasan:**
- Satu sampel dengan RestingBP <= 0 dihapus karena nilai tekanan darah tidak mungkin 0 atau negatif, dan menghapus satu sampel tidak akan berdampak signifikan pada keseluruhan dataset.
- Nilai Cholesterol = 0 diganti dengan nilai median dari sampel yang valid, karena nilai kolesterol tidak mungkin 0 dan median dipilih karena lebih tahan terhadap outlier dibandingkan mean.
- Jika terdapat nilai Oldpeak negatif, diganti dengan 0 karena secara klinis, depresi ST yang negatif dapat diinterpretasikan sebagai tidak ada depresi (0).

### 2. Feature Engineering: Transformasi Oldpeak

```python
# Transformasi logaritmik pada Oldpeak
df_clean['Oldpeak'] = np.log1p(df_clean['Oldpeak'])
```

**Alasan:**
- Variabel Oldpeak menunjukkan distribusi yang miring ke kanan (right-skewed). Transformasi logaritmik (log(x+1) untuk menghindari log(0)) dilakukan untuk membuat distribusi lebih mendekati normal, yang dapat membantu meningkatkan performa algoritma machine learning, terutama yang mengasumsikan distribusi normal seperti Logistic Regression.

### 3. Encoding Fitur Kategorikal

```python
# Label Encoding untuk fitur biner
label_encoder = LabelEncoder()
df_clean['Sex'] = label_encoder.fit_transform(df_clean['Sex'])  # M=1, F=0
df_clean['ExerciseAngina'] = label_encoder.fit_transform(df_clean['ExerciseAngina'])  # Y=1, N=0

# One-Hot Encoding untuk fitur kategorikal non-ordinal
df_clean = pd.get_dummies(df_clean, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)
```

**Alasan:**
- Label Encoding digunakan untuk fitur biner (Sex dan ExerciseAngina) karena hanya memiliki dua nilai yang dapat direpresentasikan sebagai 0 dan 1.
- One-Hot Encoding diterapkan pada fitur kategorikal non-ordinal (ChestPainType, RestingECG, dan ST_Slope) karena tidak ada hubungan ordinal antar kategori. One-Hot Encoding mencegah algoritma menafsirkan urutan yang tidak ada antar kategori.
- Parameter drop_first=True digunakan untuk menghindari multikolinearitas pada fitur hasil encoding (dummy variable trap).

### 4. Scaling Fitur Numerik

```python
# Scaling fitur numerik
scaler = StandardScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
```

**Alasan:**
- Standard Scaling dilakukan pada fitur numerik agar semua fitur memiliki mean = 0 dan standar deviasi = 1. Ini penting terutama untuk algoritma seperti Logistic Regression yang sensitif terhadap skala fitur.
- Scaling membantu mempercepat konvergensi algoritma gradient descent dan mencegah fitur dengan skala besar mendominasi fitur dengan skala kecil dalam perhitungan jarak (penting untuk algoritma berbasis jarak).

### 5. Pemisahan Data

```python
# Memisahkan fitur dan target
X = df_clean.drop('HeartDisease', axis=1)
y = df_clean['HeartDisease']

# Pemisahan data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

**Alasan:**
- Data dibagi menjadi set latih (80%) dan set uji (20%), yang merupakan pembagian standar dalam machine learning.
- Parameter stratify=y memastikan bahwa distribusi kelas target (HeartDisease) tetap sama di kedua set data, yang penting untuk menjaga representasi kelas dalam data latih dan uji.

### 6. Penyetaraan Distribusi Kelas dengan SMOTE

```python
# Inisialisasi SMOTE
smote = SMOTE(random_state=42)

# Menerapkan SMOTE pada data latih
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Alasan:**
- SMOTE (Synthetic Minority Oversampling Technique) diterapkan untuk menyetarakan distribusi kelas pada data latih.
- Meskipun ketidakseimbangan kelas dalam dataset ini tidak terlalu ekstrem, SMOTE tetap diterapkan untuk memastikan model dapat belajar dengan baik dari kedua kelas, terutama karena dalam konteks medis, false negative memiliki konsekuensi yang lebih serius.
- SMOTE hanya diterapkan pada data latih untuk mencegah data leakage, sementara data uji tetap mencerminkan distribusi kelas alami untuk evaluasi yang realistis.

## Modeling

Pada tahap ini, tiga algoritma machine learning diimplementasikan dan dibandingkan untuk menemukan model terbaik dalam memprediksi penyakit jantung: Logistic Regression, Random Forest Classifier, dan Gradient Boosting Classifier.

### 1. Penjelasan Algoritma yang Digunakan

#### Logistic Regression

**Kelebihan:**
- Model yang sederhana dan interpretable, memungkinkan analisis pengaruh masing-masing fitur.
- Baik untuk klasifikasi biner dan efisien secara komputasional.
- Memberikan probabilitas output yang terkalibrasi dengan baik.

**Kekurangan:**
- Berasumsi bahwa hubungan antar fitur dan target bersifat linier.
- Tidak dapat menangkap interaksi kompleks antar fitur tanpa penambahan fitur interaksi secara manual.
- Rentan terhadap outlier dan multikolinearitas.

#### Random Forest Classifier

**Kelebihan:**
- Model ensemble berbasis pohon keputusan yang tahan terhadap overfitting.
- Dapat menangkap hubungan non-linear dan interaksi kompleks antar fitur.
- Memberikan informasi pentingnya fitur (feature importance).
- Tidak memerlukan asumsi distribusi fitur.

**Kekurangan:**
- Lebih kompleks dan membutuhkan lebih banyak sumber daya komputasi.
- Kurang interpretable dibandingkan Logistic Regression.
- Probabilitas output tidak selalu terkalibrasi dengan baik.

#### Gradient Boosting Classifier

**Kelebihan:**
- Model ensemble yang sangat kuat untuk menangkap pola kompleks.
- Sering kali memberikan performa terbaik pada berbagai dataset.
- Dapat menangani hubungan non-linear dengan baik.

**Kekurangan:**
- Lebih rentan terhadap overfitting dibandingkan Random Forest.
- Komputasional lebih intensif dan membutuhkan waktu pelatihan lebih lama.
- Parameter yang sensitif memerlukan tuning yang cermat.
- Kurang interpretable dibandingkan model sederhana.

### 2. Implementasi Baseline Model

```python
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Melatih dan mengevaluasi setiap model
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='f1')
    results[name] = evaluate_model(model, X_test, y_test)
```

Untuk masing-masing model, cross-validation dengan 5-fold dilakukan untuk mengevaluasi stabilitas performa model, menggunakan F1-score sebagai metrik evaluasi utama.

### 3. Perbandingan Performa Model Baseline

Berdasarkan hasil evaluasi, perbandingan performa ketiga model adalah sebagai berikut:

| Model                 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression   | 0.8750   | 0.8834    | 0.8921 | 0.8878   | 0.9323  |
| Random Forest         | 0.8750   | 0.8834    | 0.8921 | 0.8878   | 0.9304  |
| Gradient Boosting     | 0.8586   | 0.8877    | 0.8529 | 0.8700   | 0.9253  |

Dari hasil perbandingan, **Logistic Regression** menunjukkan performa terbaik dengan F1-score tertinggi (0.8878) dan ROC-AUC tertinggi (0.9323). Meskipun secara teori model ensemble seperti Random Forest dan Gradient Boosting dapat menangkap pola kompleks lebih baik, dataset ini mungkin memiliki hubungan yang relatif linier antara fitur dan target, yang menjadikan Logistic Regression algoritma yang paling sesuai.

### 4. Pemilihan dan Improvement Model

Berdasarkan hasil baseline, **Logistic Regression** dipilih sebagai model terbaik dan dilakukan hyperparameter tuning untuk meningkatkan performa lebih lanjut.

```python
# Definisi grid hyperparameter untuk Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Inisialisasi RandomizedSearchCV
tuned_model = RandomizedSearchCV(
    estimator=models['Logistic Regression'],
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

# Pelatihan model
tuned_model.fit(X_train_balanced, y_train_balanced)
```

**Proses Hyperparameter Tuning:**
- Parameter `C`: Mengontrol kekuatan regularisasi (nilai C yang lebih kecil berarti regularisasi lebih kuat). Range dari 0.01 hingga 100 digunakan untuk mencoba berbagai tingkat regularisasi.
- Parameter `penalty`: Menentukan jenis regularisasi (L1 atau L2). L1 dapat melakukan feature selection, sementara L2 cenderung mengurangi koefisien yang besar.
- Parameter `solver`: Algoritma untuk optimasi. 'liblinear' dipilih karena efisien untuk dataset kecil hingga menengah dan mendukung keduanya L1 dan L2.

**Hasil Tuning:**
Parameter terbaik yang ditemukan adalah `C=10`, `penalty='l2'`, dan `solver='liblinear'`. Model dengan parameter ini memberikan peningkatan performa dibandingkan model baseline.

## Evaluation

Evaluasi model dilakukan menggunakan beberapa metrik untuk memastikan performa yang komprehensif. Berikut adalah penjelasan metrik yang digunakan:

### 1. Metrik Evaluasi

#### Accuracy
Mengukur proporsi prediksi yang benar (baik positif maupun negatif) dari total prediksi.

**Formula:**

Accuracy = (TP + TN)/(TP + TN + FP + FN)

Dimana:
- TP (True Positive): Jumlah pasien dengan penyakit jantung yang diprediksi benar
- TN (True Negative): Jumlah pasien tanpa penyakit jantung yang diprediksi benar
- FP (False Positive): Jumlah pasien tanpa penyakit jantung yang diprediksi memiliki penyakit
- FN (False Negative): Jumlah pasien dengan penyakit jantung yang diprediksi tidak memiliki penyakit

#### Precision
Mengukur proporsi prediksi positif yang benar dari semua prediksi positif. Penting untuk menghindari false positive (mendiagnosis penyakit pada pasien sehat).

**Formula:**

Precision = TP/(TP + FP)

#### Recall (Sensitivity)
Mengukur proporsi kasus positif aktual yang berhasil diprediksi. Penting untuk menghindari false negative (tidak mendiagnosis penyakit pada pasien yang sebenarnya sakit).

**Formula:** 

Recall = TP/(TP + FN)

#### F1-Score
Rata-rata harmonik antara precision dan recall, memberikan satu metrik yang menyeimbangkan keduanya. Penting ketika ingin keseimbangan antara mendeteksi semua kasus positif dan menghindari false positive.

**Formula:**

F1-Score = 2 × (Precision × Recall)/(Precision + Recall)

#### ROC-AUC
Area Under the Receiver Operating Characteristic Curve, mengukur kemampuan model untuk membedakan antara kelas positif dan negatif. Nilai 1 berarti klasifikasi sempurna, 0.5 berarti klasifikasi acak. ROC-AUC tidak dipengaruhi oleh threshold klasifikasi, sehingga memberikan pandangan yang lebih komprehensif tentang performa model.

### 2. Hasil Evaluasi Model Terbaik

Setelah hyperparameter tuning, model Logistic Regression terbaik menunjukkan hasil evaluasi berikut:

- Accuracy: 0.8804
- Precision: 0.8922
- Recall: 0.8922
- F1-Score: 0.8922
- ROC-AUC: 0.9321

Confusion Matrix menunjukkan distribusi prediksi sebagai berikut:
- True Negative: 72
- False Positive: 11
- False Negative: 11
- True Positive: 91

**Interpretasi Hasil:**
- Model menghasilkan accuracy yang tinggi (88.04%), yang berarti model dapat memprediksi dengan benar lebih dari 88% dari semua kasus.
- Precision sebesar 89.22% menunjukkan bahwa dari semua pasien yang diprediksi memiliki penyakit jantung, hampir 90% benar-benar memiliki penyakit tersebut.
- Recall sebesar 89.22% menunjukkan bahwa model berhasil mengidentifikasi lebih dari 90% dari semua pasien yang benar-benar memiliki penyakit jantung.
- F1-Score sebesar 89.22% menunjukkan keseimbangan yang baik antara precision dan recall.
- ROC-AUC sebesar 93.21% mengindikasikan bahwa model memiliki kemampuan yang sangat baik dalam membedakan antara pasien dengan dan tanpa penyakit jantung.

Kurva ROC menunjukkan hubungan antara True Positive Rate dan False Positive Rate pada berbagai threshold, dengan area di bawah kurva (AUC) mencapai 0.9321, yang mengindikasikan performa model yang sangat baik.

## Kesimpulan

Proyek ini berhasil mengembangkan model machine learning untuk memprediksi keberadaan penyakit jantung berdasarkan parameter kesehatan pasien. Setelah membandingkan tiga algoritma (Logistic Regression, Random Forest, dan Gradient Boosting), Logistic Regression terbukti memberikan performa terbaik dengan F1-score 0.8878 pada model baseline.

Melalui hyperparameter tuning, performa model Logistic Regression berhasil ditingkatkan, mencapai F1-score 0.8922 dan ROC-AUC 0.9321 pada data uji. Hasil ini menunjukkan bahwa model cukup meyakinkan dalam membedakan pasien dengan dan tanpa penyakit jantung.

Beberapa temuan penting dari proyek ini:
1. Fitur seperti jenis nyeri dada (terutama Asymptomatic), adanya angina saat berolahraga, dan kemiringan segmen ST memiliki hubungan kuat dengan keberadaan penyakit jantung.
2. Meskipun model ensemble seperti Random Forest dan Gradient Boosting lebih kompleks, Logistic Regression yang lebih sederhana memberikan performa terbaik, menunjukkan bahwa kompleksitas model tidak selalu menghasilkan performa yang lebih baik.
3. Penanganan data yang tepat, termasuk imputasi nilai yang tidak valid dan penyetaraan distribusi kelas, terbukti penting dalam meningkatkan performa model.

Model ini dapat berfungsi sebagai alat skrining awal yang hemat biaya untuk membantu tenaga medis mengidentifikasi pasien berisiko tinggi penyakit jantung, terutama di fasilitas kesehatan dengan sumber daya terbatas. Namun, untuk implementasi klinis, model perlu divalidasi lebih lanjut dengan dataset eksternal yang lebih beragam dan diuji dalam lingkungan klinis yang sebenarnya.

Untuk pengembangan masa depan, model dapat ditingkatkan dengan:
1. Menambahkan fitur tambahan seperti riwayat keluarga, indeks massa tubuh, dan gaya hidup.
2. Mengeksplorasi model yang lebih kompleks jika dataset yang lebih besar tersedia.
3. Mengembangkan interpretabilitas model untuk memberikan wawasan klinis yang lebih bermanfaat.
4. Membangun antarmuka pengguna yang ramah untuk memudahkan penggunaan oleh tenaga medis.

## Referensi

[1] World Health Organization. "Cardiovascular diseases (CVDs)." *WHO*, 2021, https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds).

[2] Kementerian Kesehatan RI. "Hasil Utama Riskesdas 2018." *Badan Penelitian dan Pengembangan Kesehatan*, 2018.

[3] Benjamin, E. J., et al. "Heart Disease and Stroke Statistics—2019 Update: A Report From the American Heart Association." *Circulation*, vol. 139, no. 10, 2019, pp. e56-e528.

[4] Shah, D., et al. "A Comprehensive Machine Learning Approach for Predicting Cardiovascular Disease." *Journal of Biomedical Informatics*, vol. 108, 2020, p. 103493.

[5] Mohan, S., et al. "Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques." *IEEE Access*, vol. 7, 2019, pp. 81542-81554.
