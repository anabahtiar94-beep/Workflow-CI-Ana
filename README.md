# Workflow-CI (Ana) — Heart Disease Prediction

Workflow CI untuk re-training model **Heart Disease Prediction** secara otomatis menggunakan MLflow Project + GitHub Actions, dan mem-build/push Docker image ke Docker Hub.

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml
├── MLProject/
│   ├── MLProject                          # MLflow Project spec
│   ├── conda.yaml                         # environment
│   ├── modelling.py                       # skrip training untuk CI
│   └── heart_disease_preprocessing/       # data siap latih (train.csv, test.csv)
└── README.md
```

## Sebelum push ke GitHub

1. Buat repository baru **Public** bernama `Workflow-CI-Ana` di akun GitHub [anabahtiar94-beep](https://github.com/anabahtiar94-beep).
2. Tambahkan GitHub Actions secrets pada repo (Settings → Secrets and variables → Actions):
   - `DOCKERHUB_USERNAME` → username Docker Hub Anda (huruf kecil semua).
   - `DOCKERHUB_TOKEN` → Access Token dari Docker Hub (Account Settings → Security → New Access Token), **bukan password**.
3. Push seluruh isi folder ini ke repo tersebut.

## Menjalankan secara lokal (opsional, untuk uji coba sebelum push)

```bash
cd MLProject
pip install mlflow==2.19.0 pandas numpy scikit-learn
mlflow run . --env-manager=local
```


