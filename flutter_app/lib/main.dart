// Delivery Time Predictor — Flutter front-end.
//
// This is the "your edge activates here" part of the playbook: a real mobile app
// that calls YOUR trained model over HTTP. It talks to the FastAPI /predict endpoint.
//
// Setup:
//   1. Start the API:   uvicorn api.main:app --reload   (from the project root)
//   2. Point `apiBaseUrl` below at your machine:
//        - iOS simulator / macOS / web:  http://127.0.0.1:8000
//        - Android emulator:             http://10.0.2.2:8000   (special alias)
//        - real phone:                   http://<your-computer-LAN-IP>:8000
//   3. flutter run
//
// The whole ML-on-mobile idea in one screen: fill a form -> POST JSON -> show the
// model's answer. Nothing here is Flutter-exotic; the interesting part is that the
// number comes from a model YOU trained.

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// CHANGE THIS to match how you're running the app (see the note above).
const String apiBaseUrl = 'http://127.0.0.1:8000';

void main() => runApp(const DeliveryApp());

class DeliveryApp extends StatelessWidget {
  const DeliveryApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Delivery Time Predictor',
      theme: ThemeData(
        colorSchemeSeed: Colors.deepOrange,
        useMaterial3: true,
      ),
      home: const PredictScreen(),
    );
  }
}

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  // The few inputs a user actually chooses. Everything else uses the API's
  // sensible defaults (age, ratings, coordinates, etc.).
  String _traffic = 'Medium';
  String _weather = 'Sunny';
  String _festival = 'No';
  double _distanceProxyKm = 5; // we fake a delivery point this far from the restaurant

  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _result;

  static const _trafficOptions = ['Low', 'Medium', 'High', 'Jam'];
  static const _weatherOptions = [
    'Sunny', 'Cloudy', 'Fog', 'Windy', 'Stormy', 'Sandstorms'
  ];

  Future<void> _predict() async {
    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });

    // A fixed restaurant in Bangalore; we offset the delivery point north by
    // roughly `_distanceProxyKm` (1 degree lat ~ 111 km) so the slider changes
    // the haversine distance the model sees.
    const restLat = 12.9716, restLng = 77.5946;
    final delLat = restLat + (_distanceProxyKm / 111.0);

    final body = {
      'Restaurant_latitude': restLat,
      'Restaurant_longitude': restLng,
      'Delivery_location_latitude': delLat,
      'Delivery_location_longitude': restLng,
      'Road_traffic_density': _traffic,
      'Weatherconditions': _weather,
      'Festival': _festival,
    };

    try {
      final resp = await http.post(
        Uri.parse('$apiBaseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );
      if (resp.statusCode == 200) {
        setState(() => _result = jsonDecode(resp.body) as Map<String, dynamic>);
      } else {
        setState(() => _error = 'API error ${resp.statusCode}: ${resp.body}');
      }
    } catch (e) {
      setState(() => _error = 'Could not reach API at $apiBaseUrl\n$e');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Delivery Time Predictor')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _dropdown('Traffic', _traffic, _trafficOptions,
              (v) => setState(() => _traffic = v!)),
          _dropdown('Weather', _weather, _weatherOptions,
              (v) => setState(() => _weather = v!)),
          _dropdown('Festival day?', _festival, const ['No', 'Yes'],
              (v) => setState(() => _festival = v!)),
          const SizedBox(height: 12),
          Text('Distance: ${_distanceProxyKm.toStringAsFixed(1)} km'),
          Slider(
            value: _distanceProxyKm,
            min: 1,
            max: 20,
            divisions: 19,
            label: '${_distanceProxyKm.toStringAsFixed(0)} km',
            onChanged: (v) => setState(() => _distanceProxyKm = v),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: _loading ? null : _predict,
            icon: _loading
                ? const SizedBox(
                    width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.delivery_dining),
            label: Text(_loading ? 'Predicting…' : 'Predict delivery time'),
          ),
          const SizedBox(height: 24),
          if (_error != null)
            Card(
              color: Colors.red.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(_error!, style: TextStyle(color: Colors.red.shade900)),
              ),
            ),
          if (_result != null) _resultCard(_result!),
        ],
      ),
    );
  }

  Widget _dropdown(String label, String value, List<String> options,
      ValueChanged<String?> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: InputDecorator(
        decoration: InputDecoration(labelText: label, border: const OutlineInputBorder()),
        child: DropdownButtonHideUnderline(
          child: DropdownButton<String>(
            value: value,
            isExpanded: true,
            items: options
                .map((o) => DropdownMenuItem(value: o, child: Text(o)))
                .toList(),
            onChanged: onChanged,
          ),
        ),
      ),
    );
  }

  Widget _resultCard(Map<String, dynamic> r) {
    final minutes = (r['predicted_minutes'] as num).toDouble();
    final lateProb = (r['late_probability'] as num).toDouble();
    final willBeLate = r['will_be_late'] as bool;
    return Card(
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('${minutes.toStringAsFixed(0)} min',
                style: Theme.of(context).textTheme.displaySmall),
            const Text('estimated delivery time'),
            const Divider(height: 28),
            Row(
              children: [
                Icon(willBeLate ? Icons.warning_amber : Icons.check_circle,
                    color: willBeLate ? Colors.orange : Colors.green),
                const SizedBox(width: 8),
                Text('${(lateProb * 100).toStringAsFixed(0)}% chance of being late'),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
