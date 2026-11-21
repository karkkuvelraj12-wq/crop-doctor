import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CropDoctorApp());
}

class CropDoctorApp extends StatelessWidget {
  const CropDoctorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Crop Doctor',
      theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
      home: const HomePage(),
    );
  }
}

// ---------------- LANGUAGES ----------------

const Map<String, String> languages = {
  'en': 'English',
  'ta': 'தமிழ்',
  'hi': 'हिन्दी',
};

const Map<String, String> ttsLanguages = {
  'en': 'en-IN',
  'ta': 'ta-IN',
  'hi': 'hi-IN',
};

// ---------------- ADVICE (offline) ----------------

const Map<String, Map<String, String>> advice = {
  'retake': {
    'en': 'Low confidence. Please retake a clear, close-up photo of a single leaf.',
    'ta': 'நம்பிக்கை குறைவு. ஒரு இலையின் தெளிவான நெருக்கமான புகைப்படத்தை மீண்டும் எடுக்கவும்.',
    'hi': 'विश्वास कम है। कृपया एक पत्ती की साफ और नज़दीकी तस्वीर दोबारा लें।',
  },
  'healthy': {
    'en': 'The plant appears healthy. Continue monitoring it regularly.',
    'ta': 'செடி ஆரோக்கியமாகத் தெரிகிறது. தொடர்ந்து கண்காணிக்கவும்.',
    'hi': 'पौधा स्वस्थ दिखाई देता है। नियमित रूप से निगरानी करें।',
  },
  'fungal': {
    'en': 'Possible fungal disease. Remove infected leaves, avoid watering from above, and maintain proper airflow. Consult an agriculture officer for treatment.',
    'ta': 'பூஞ்சை நோய் இருக்கலாம். பாதிக்கப்பட்ட இலைகளை அகற்றவும். மேலிருந்து நீர் பாய்ச்சுவதைத் தவிர்க்கவும். சரியான காற்றோட்டத்தை பராமரிக்கவும். சிகிச்சைக்கு வேளாண் அலுவலரை அணுகவும்.',
    'hi': 'संभावित फफूंद रोग। संक्रमित पत्तियों को हटाएं, ऊपर से पानी देने से बचें और उचित हवा का प्रवाह बनाए रखें। उपचार के लिए कृषि अधिकारी से सलाह लें।',
  },
  'bacterial': {
    'en': 'Possible bacterial disease. Remove infected leaves, use clean tools, and consult your local agriculture officer.',
    'ta': 'பாக்டீரியா நோய் இருக்கலாம். பாதிக்கப்பட்ட இலைகளை அகற்றி சுத்தமான கருவிகளைப் பயன்படுத்தவும். வேளாண் அலுவலரை அணுகவும்.',
    'hi': 'संभावित जीवाणु रोग। संक्रमित पत्तियों को हटाएं, साफ औजारों का उपयोग करें और कृषि अधिकारी से सलाह लें।',
  },
  'default': {
    'en': 'A possible plant disease was detected. Isolate the plant and consult your local agriculture officer.',
    'ta': 'தாவரத்தில் நோய் இருக்கலாம். செடியைத் தனியாக வைத்து வேளாண் அலுவலரை அணுகவும்.',
    'hi': 'पौधे में संभावित बीमारी पाई गई है। पौधे को अलग रखें और कृषि अधिकारी से सलाह लें।',
  },
};

String getAdviceKey(String label) {
  final value = label.toLowerCase();
  if (value.contains('healthy')) return 'healthy';
  if (value.contains('bacterial')) return 'bacterial';
  const fungalKeywords = [
    'blight', 'rust', 'scab', 'mildew', 'mold', 'rot', 'spot', 'measles', 'scorch',
  ];
  if (fungalKeywords.any((w) => value.contains(w))) return 'fungal';
  return 'default';
}

String formatLabel(String label) {
  return label
      .replaceAll('___', ' - ')
      .replaceAll('_', ' ')
      .replaceAll(',', '')
      .trim();
}

// ---------------- HOME PAGE ----------------

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const int imageSize = 224;

  // Below this the result is shown as "low confidence" (best guess + retake tip).
  static const double confidenceThreshold = 0.50;

  Interpreter? interpreter;
  final ImagePicker picker = ImagePicker();
  final FlutterTts tts = FlutterTts();

  List<String> labels = [];
  File? selectedImage;

  String selectedLanguage = 'en';
  String? predictedDisease;
  String? errorMessage;
  double confidence = 0;
  List<MapEntry<String, double>> topPredictions = [];

  bool isLoading = true;
  bool isProcessing = false;
  bool isUncertain = false;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  // ---------------- LOAD MODEL ----------------

  Future<void> loadModel() async {
    try {
      final loaded = await Interpreter.fromAsset('assets/crop_disease.tflite');
      final raw = await rootBundle.loadString('assets/labels.txt');
      final loadedLabels = raw
          .split('\n')
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty)
          .toList();

      final inputShape = loaded.getInputTensor(0).shape;
      final outputShape = loaded.getOutputTensor(0).shape;
      debugPrint('Input shape: $inputShape  Output shape: $outputShape');
      debugPrint('Input type: ${loaded.getInputTensor(0).type}');
      debugPrint('Labels: ${loadedLabels.length}');

      if (loadedLabels.length != 38) {
        throw Exception('Expected 38 labels, found ${loadedLabels.length}');
      }
      if (inputShape.length != 4 ||
          inputShape[1] != imageSize ||
          inputShape[2] != imageSize ||
          inputShape[3] != 3) {
        throw Exception('Unexpected model input shape: $inputShape');
      }
      if (outputShape.length != 2 || outputShape[1] != loadedLabels.length) {
        throw Exception('Unexpected model output shape: $outputShape');
      }

      interpreter = loaded;
      labels = loadedLabels;
      if (!mounted) return;
      setState(() {
        isLoading = false;
        errorMessage = null;
      });
    } catch (e) {
      debugPrint('MODEL LOADING ERROR: $e');
      if (!mounted) return;
      setState(() {
        isLoading = false;
        errorMessage = 'Model loading failed: $e';
      });
    }
  }

  // ---------------- PICK IMAGE ----------------

  Future<void> pickImage(ImageSource source) async {
    if (isProcessing || interpreter == null) return;
    try {
      final XFile? image = await picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 95,
      );
      if (image == null) return;

      setState(() {
        selectedImage = File(image.path);
        isProcessing = true;
        predictedDisease = null;
        confidence = 0;
        isUncertain = false;
        errorMessage = null;
        topPredictions = [];
      });

      await classifyImage(File(image.path));
    } catch (e) {
      showError('Image selection failed: $e');
    }
  }

  // ---------------- CLASSIFY ----------------

  Future<void> classifyImage(File file) async {
    try {
      if (interpreter == null || labels.isEmpty) {
        throw Exception('Model is not ready.');
      }

      final bytes = await file.readAsBytes();
      img.Image? decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Unable to decode image.');
      decoded = img.bakeOrientation(decoded);

      final resized = img.copyResize(
        decoded,
        width: imageSize,
        height: imageSize,
        interpolation: img.Interpolation.linear,
      );

      // Model expects float32 values in 0-255. Do NOT divide by 255.
      final List<List<List<List<double>>>> input = [
        List.generate(
          imageSize,
          (y) => List.generate(imageSize, (x) {
            final p = resized.getPixel(x, y);
            return [p.r.toDouble(), p.g.toDouble(), p.b.toDouble()];
          }),
        ),
      ];

      final List<List<double>> output = [
        List<double>.filled(labels.length, 0.0),
      ];

      interpreter!.run(input, output);
      final probs = output[0];

      final ranked = List<int>.generate(probs.length, (i) => i)
        ..sort((a, b) => probs[b].compareTo(probs[a]));

      final top3 = ranked
          .take(3)
          .map((i) => MapEntry(labels[i], probs[i]))
          .toList();

      debugPrint('----- PREDICTION RESULTS -----');
      for (final e in ranked.take(5)) {
        debugPrint('${labels[e]}: ${(probs[e] * 100).toStringAsFixed(2)}%');
      }

      final bestConfidence = probs[ranked.first];

      if (!mounted) return;
      setState(() {
        predictedDisease = labels[ranked.first];
        confidence = bestConfidence;
        topPredictions = top3;
        isUncertain = bestConfidence < confidenceThreshold;
        isProcessing = false;
      });

      await speakResult();
    } catch (e) {
      debugPrint('PREDICTION ERROR: $e');
      showError('Prediction failed: $e');
    }
  }

  void showError(String message) {
    if (!mounted) return;
    setState(() {
      isProcessing = false;
      errorMessage = message;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  // ---------------- ADVICE TEXT ----------------

  String get diseaseAdvice {
    if (predictedDisease == null) return '';
    return advice[getAdviceKey(predictedDisease!)]![selectedLanguage]!;
  }

  String get retakeAdvice => advice['retake']![selectedLanguage]!;

  // ---------------- TEXT TO SPEECH ----------------

  Future<void> speakResult() async {
    if (predictedDisease == null) return;
    try {
      await tts.stop();
      await tts.setLanguage(ttsLanguages[selectedLanguage]!);
      await tts.setSpeechRate(0.45);

      final label = formatLabel(predictedDisease!);
      final text = isUncertain
          ? '$retakeAdvice $label.'
          : '$label. $diseaseAdvice';
      await tts.speak(text);
    } catch (e) {
      debugPrint('TTS ERROR: $e');
    }
  }

  // ---------------- UI ----------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Crop Doctor 🌿',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.green.shade100,
        actions: [
          DropdownButton<String>(
            value: selectedLanguage,
            underline: const SizedBox(),
            items: languages.entries
                .map((e) => DropdownMenuItem<String>(
                      value: e.key,
                      child: Text(e.value),
                    ))
                .toList(),
            onChanged: (value) {
              if (value == null) return;
              setState(() => selectedLanguage = value);
              speakResult();
            },
          ),
          const SizedBox(width: 12),
        ],
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : errorMessage != null && interpreter == null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.error_outline,
                            color: Colors.red, size: 55),
                        const SizedBox(height: 15),
                        Text(errorMessage!, textAlign: TextAlign.center),
                        const SizedBox(height: 15),
                        ElevatedButton(
                          onPressed: () {
                            setState(() {
                              isLoading = true;
                              errorMessage = null;
                            });
                            loadModel();
                          },
                          child: const Text('Retry'),
                        ),
                      ],
                    ),
                  ),
                )
              : buildMainContent(),
    );
  }

  Widget buildMainContent() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const Text(
          'Plant Disease Detection',
          style: TextStyle(fontSize: 23, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 6),
        const Text(
          'Take or select a close-up photo of a single leaf.',
          style: TextStyle(fontSize: 14),
        ),
        const SizedBox(height: 20),

        // Image preview
        Container(
          height: 260,
          decoration: BoxDecoration(
            color: Colors.green.shade50,
            borderRadius: BorderRadius.circular(18),
          ),
          child: selectedImage == null
              ? const Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.eco, size: 65, color: Colors.green),
                      SizedBox(height: 10),
                      Text('Take a photo of a leaf'),
                    ],
                  ),
                )
              : ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: Image.file(
                    selectedImage!,
                    width: double.infinity,
                    height: 260,
                    fit: BoxFit.contain,
                  ),
                ),
        ),
        const SizedBox(height: 20),

        // Buttons
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed:
                    isProcessing ? null : () => pickImage(ImageSource.camera),
                icon: const Icon(Icons.camera_alt),
                label: const Text('Camera'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: OutlinedButton.icon(
                onPressed:
                    isProcessing ? null : () => pickImage(ImageSource.gallery),
                icon: const Icon(Icons.photo),
                label: const Text('Gallery'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 20),

        if (isProcessing)
          const Column(
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 10),
              Text('Analyzing leaf image...'),
            ],
          ),

        if (errorMessage != null && interpreter != null)
          Card(
            color: Colors.red.shade50,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Text(errorMessage!,
                  style: const TextStyle(color: Colors.red)),
            ),
          ),

        // Result (best guess is ALWAYS shown)
        if (predictedDisease != null && !isProcessing)
          Card(
            color: isUncertain ? Colors.orange.shade50 : Colors.green.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    isUncertain
                        ? 'Low Confidence - Best Guess'
                        : 'Detection Result',
                    style: const TextStyle(
                        fontSize: 19, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    formatLabel(predictedDisease!),
                    style: const TextStyle(
                        fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
                    style: const TextStyle(fontSize: 16),
                  ),
                  const SizedBox(height: 4),
                  LinearProgressIndicator(
                    value: confidence.clamp(0.0, 1.0),
                    minHeight: 6,
                    color: isUncertain ? Colors.orange : Colors.green,
                  ),
                  const SizedBox(height: 12),
                  if (isUncertain) ...[
                    Text(
                      retakeAdvice,
                      style: const TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 8),
                  ],
                  Text(diseaseAdvice, style: const TextStyle(fontSize: 15)),
                  const SizedBox(height: 12),
                  const Text('Other possibilities:',
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  ...topPredictions.skip(1).map(
                        (e) => Text(
                          '${formatLabel(e.key)} - ${(e.value * 100).toStringAsFixed(1)}%',
                          style: const TextStyle(fontSize: 13),
                        ),
                      ),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: speakResult,
                    icon: const Icon(Icons.volume_up),
                    label: const Text('Listen'),
                  ),
                ],
              ),
            ),
          ),

        const SizedBox(height: 20),
        const Text(
          'Note: Predictions are model estimates, not confirmed diagnoses. '
          'The model recognizes only the 38 PlantVillage classes (14 crops) and '
          'may be less accurate on real-world photos.',
          style: TextStyle(fontSize: 12, color: Colors.grey),
        ),
      ],
    );
  }

  @override
  void dispose() {
    interpreter?.close();
    tts.stop();
    super.dispose();
  }
}