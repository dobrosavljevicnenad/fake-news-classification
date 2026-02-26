# WELFake Data Mining Project  
Istraživanje podataka 2 – Matematički fakultet  

Autor: Nenad Dobrosavljević  

---

## Opis projekta

U ovom projektu izvršena je analiza skupa podataka **WELFake** primenom tri metode istraživanja podataka:

- Klasifikacija
- Klasterovanje
- Pravila pridruživanja

Cilj rada je poređenje performansi različitih algoritama i analiza strukture tekstualnih podataka koji predstavljaju realne i lažne vesti.

---

## Preuzimanje skupa podataka

Skup podataka nije uključen u repozitorijum zbog svoje veličine.

Potrebno je preuzeti dataset sa sledeće adrese:

https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Naziv fajla:

WELFake_Dataset.csv


Nakon preuzimanja, fajl je potrebno postaviti u sledeći direktorijum:


data/raw/


---

# Redosled pokretanja

Notebook fajlove je potrebno pokretati sledećim redosledom:


## 01_eda.ipynb

Eksplorativna analiza podataka i priprema karakteristika:

- Učitavanje podataka  
- Čišćenje i normalizacija teksta  
- TF-IDF reprezentacija  
- Smanjenje dimenzionalnosti primenom SVD metode  
- 2D i 3D vizualizacija  


## 02_classification.ipynb

Nadzirano učenje i evaluacija modela:

- Podela na trening i test skup  
- Treniranje pet klasifikacionih modela  
- Evaluacija performansi modela  
- Poređenje sledećih reprezentacija:
  - FULL reprezentacija  
  - SELECT reprezentacija  
  - SVD reprezentacija  


## 03_clustering.ipynb

Nenadzirano učenje:

- Uzorkovanje 10.000 instanci  
- Određivanje optimalnog broja klastera  
- Evaluacija više algoritama klasterovanja  
- Uporedna analiza performansi  


## 04_association_rules.ipynb

Otkrivanje asocijativnih pravila:

- Transformacija podataka u transakcioni format  
- Primena Apriori algoritma  
- Primena FP-Growth algoritma  
- Poređenje Fake i Real podskupa

---