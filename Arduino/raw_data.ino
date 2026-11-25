const int SENSOR_PINS[5] = {A0, A1, A2, A3, A4};
float baseline[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
const int BAUD_RATE = 9600;
const int SAMPLE_DELAY_MS = 20;

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial);

  Serial.println("STATUS:CALIBRATING");

  int numSamples = 50;
  for (int i = 0; i < numSamples; i++) {
    for (int s = 0; s < 5; s++) {
      baseline[s] += analogRead(SENSOR_PINS[s]);
    }
    delay(20);
  }

  for (int s = 0; s < 5; s++) {
    baseline[s] /= (float)numSamples;
  }

  Serial.println("STATUS:READY");
}

void loop() {
  float calibratedValues[5];

  for (int s = 0; s < 5; s++) {
    float raw = analogRead(SENSOR_PINS[s]);
    calibratedValues[s] = raw - baseline[s];
  }

  Serial.print("DATA:");

  for (int s = 0; s < 5; s++) {
    Serial.print(calibratedValues[s], 1);
    if (s < 4) {
      Serial.print(",");
    }
  }

  Serial.println();

  delay(SAMPLE_DELAY_MS);
}
