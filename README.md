# Crop Doctor 🌿

Offline AI crop disease detection for farmers, on a low-end Android phone.
Take a photo of a leaf and get the likely disease, a confidence score and treatment advice in **Tamil, Hindi or English**, with voice output. No internet needed.

> An agronomist in every farmer's pocket.

## Screenshots

| English | Tamil | Hindi |
|---|---|---|
| ![English](docs/shot-en.png) | ![Tamil](docs/shot-ta.png) | ![Hindi](docs/shot-hi.png) |

## Features

- **Fully offline:** the model runs on the phone (TensorFlow Lite).
- **38 classes across 14 crops** (diseases and healthy leaves), from the PlantVillage dataset.
- **Confidence score and top-3 predictions.** Below 50% confidence the app asks for a retake.
- **Local-language advice and voice:** Tamil, Hindi, English (on-device text-to-speech).
- **Advice stored locally**, so nothing depends on a server.

## How it works

Leaf photo → resize to 224×224 → MobileNetV3-Small (quantized TFLite) → softmax over 38 classes → confidence check → advice in the chosen language → spoken aloud.

## Tech stack

| Layer | Choice |
|---|---|
| Model | MobileNetV3-Small, transfer learning (TensorFlow / Keras) |
| Data | PlantVillage via `tensorflow_datasets` |
| Export | TensorFlow Lite, quantized |
| App | Flutter (Android), `tflite_flutter`, `image_picker`, `flutter_tts` |

## Results

- Test accuracy on held-out PlantVillage images: **[xx]%**
- Note: PlantVillage photos are clean lab images, so accuracy on real field photos will be lower. Field-photo validation is the next step.

## Repository structure

```
crop-doctor/
├── lib/main.dart              # Flutter app
├── assets/
│   ├── crop_disease.tflite    # trained model
│   └── labels.txt             # 38 class names
├── training/
│   └── train_crop_disease.py  # run in Google Colab (GPU)
├── docs/                      # screenshots
└── pubspec.yaml
```

## Run the app

1. Install Flutter and Android Studio, and turn on USB debugging on your phone.
2. `flutter pub get`
3. `flutter run -d <your-device-id>`

Build notes: the Android build needs JDK 17 and NDK installed. If you set `org.gradle.java.home` in `android/gradle.properties`, point it at your own JDK 17 path.

## Retrain the model

Open `training/train_crop_disease.py` in Google Colab (GPU runtime) and run it. It saves `crop_disease.tflite` and `labels.txt`. Copy both into `assets/`.

## Limitations

- Recognizes only the 38 PlantVillage classes. Other crops will give unreliable results.
- Predictions are estimates, not confirmed diagnoses. Advice is general; consult an agriculture officer.

## Roadmap

- Fine-tune on real field photos
- More crops (rice, banana, coconut) and more Indian languages
- Offline outbreak map that syncs when online

## Team

[Team name] · [Member names] · [College / city]
