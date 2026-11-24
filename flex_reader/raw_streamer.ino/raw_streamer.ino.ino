// Flex Sensor Reader with Initial Open-Hand Calibration Only
const int SENSOR_PINS[5] = {A0, A1, A2, A3, A4}; // Adjust pin numbers if needed
float baseline[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
const int BAUD_RATE = 9600;
const int SAMPLE_DELAY_MS = 20; // ~50 Hz sampling rate

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial); // Wait for serial port to connect

  Serial.println("STATUS:CALIBRATING (Keep your hand completely open)...");
  
  // Take 50 samples over ~1 second to establish the open-hand baseline
  int numSamples = 50;
  for (int i = 0; i < numSamples; i++) {
    for (int s = 0; s < 5; s++) {
      baseline[s] += analogRead(SENSOR_PINS[s]);
    }
    delay(20);
  }
  
  // Average out the baseline values
  for (int s = 0; s < 5; s++) {
    baseline[s] /= (float)numSamples;
  }

  Serial.println("STATUS:READY");
}

void loop() {
  // Read raw values and subtract the initial open-hand baseline
  float calibratedValues[5];
  for (int s = 0; s < 5; s++) {
    float raw = analogRead(SENSOR_PINS[s]);
    calibratedValues[s] = raw - baseline[s];
  }

  // Output in the exact requested format: DATA:val0,val1,val2,val3,val4
  Serial.print("DATA:");
  for (int s = 0; s < 5; s++) {
    Serial.print(calibratedValues[s], 1); // Print with 1 decimal place
    if (s < 4) {
      Serial.print(",");
    }
  }
  Serial.println();

  delay(SAMPLE_DELAY_MS);
}