/**
 * AI-Enabled Underwater Object Detection System - ESP32 Firmware
 * 
 * This firmware controls an SG90 servo motor to rotate an ultrasonic distance sensor 
 * (such as JSN-SR04T waterproof sensor or HC-SR04) back and forth from 0 to 180 degrees.
 * It takes non-blocking measurements, applies digital filtering, and streams raw JSON
 * telemetry over the USB Serial port for consumption by our Python backend and UI.
 * 
 * Hardware Connections:
 * - Servo SG90 Signal Pin -> ESP32 GPIO 18
 * - Ultrasonic Trigger Pin -> ESP32 GPIO 5
 * - Ultrasonic Echo Pin   -> ESP32 GPIO 17
 */

#include <Arduino.h>
#include <ESP32Servo.h>

// --- PIN DEFINITIONS ---
#define SERVO_PIN      18
#define TRIGGER_PIN    5
#define ECHO_PIN       17

// --- RADAR SETTINGS ---
#define MIN_ANGLE      0
#define MAX_ANGLE      180
#define STEP_DEGREE    2       // Sweep step resolution
#define SWEEP_SPEED_MS 35      // Milliseconds delay between steps (controls sweep speed)

// --- PHYSICS CONSTANTS (UNDERWATER) ---
// Speed of sound in water is ~1480 m/s compared to 343 m/s in air.
// If using JSN-SR04T underwater, we adjust calculations accordingly.
#define SOUND_SPEED_WATER_M_S 1480.0
#define MICROSECONDS_TO_METERS (SOUND_SPEED_WATER_M_S / 2000000.0)

// --- GLOBAL OBJECTS & VARIABLES ---
Servo radarServo;
int currentAngle = MIN_ANGLE;
bool sweepingForward = true;
unsigned long lastStepTime = 0;

// Filter history
#define FILTER_SIZE 5
float distanceHistory[FILTER_SIZE] = {0};
int historyIndex = 0;

// --- FUNCTION DECLARATIONS ---
float readRawDistance();
float applyMedianFilter(float newVal);
void sendTelemetry(int angle, float distance, unsigned long pulseUs);

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  // Set up ESP32 Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  radarServo.setPeriodHertz(50); // Standard 50hz servo
  radarServo.attach(SERVO_PIN, 500, 2400); // SG90 servo pulse width limits (500us to 2400us)
  
  radarServo.write(currentAngle);
  delay(500); // Allow servo to home
  
  Serial.println("{\"status\":\"system_ready\",\"message\":\"ESP32 Underwater Sonar online.\"}");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Perform a sweep step non-blockingly
  if (currentTime - lastStepTime >= SWEEP_SPEED_MS) {
    lastStepTime = currentTime;
    
    // Position servo
    radarServo.write(currentAngle);
    
    // Allow servo to settle briefly before trigger
    delay(5); 
    
    // Measure and process distance
    unsigned long pulseUs;
    float rawDist = readRawDistance();
    float filteredDist = applyMedianFilter(rawDist);
    
    // Calculate pulse duration for the signal features
    pulseUs = (unsigned long)((rawDist / SOUND_SPEED_WATER_M_S) * 2000000.0);
    
    // Stream telemetry to USB Serial port
    sendTelemetry(currentAngle, filteredDist, pulseUs);
    
    // Step angle
    if (sweepingForward) {
      currentAngle += STEP_DEGREE;
      if (currentAngle >= MAX_ANGLE) {
        currentAngle = MAX_ANGLE;
        sweepingForward = false;
      }
    } else {
      currentAngle -= STEP_DEGREE;
      if (currentAngle <= MIN_ANGLE) {
        currentAngle = MIN_ANGLE;
        sweepingForward = true;
      }
    }
  }
}

/**
 * Triggers the ultrasonic transducer and measures the echo pulse duration.
 * Return: Distance in meters, or -1.0 if out of range or timeout.
 */
float readRawDistance() {
  // Clear trigger pin
  digitalWrite(TRIGGER_PIN, LOW);
  delayMicroseconds(2);
  
  // Send a 10us HIGH pulse to trigger
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER_PIN, LOW);
  
  // Measure return echo pulse width. Timeout set to 30ms (~22m underwater depth range)
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  
  if (duration == 0) {
    return -1.0; // Out of range or timeout
  }
  
  // Distance = (Time of flight / 2) * speed of sound in water
  float distanceMeters = duration * MICROSECONDS_TO_METERS;
  return distanceMeters;
}

/**
 * Simple 5-sample Median Filter to remove spike noise and bubble/turbulent reflections.
 */
float applyMedianFilter(float newVal) {
  // Save to history ring-buffer
  distanceHistory[historyIndex] = newVal;
  historyIndex = (historyIndex + 1) % FILTER_SIZE;
  
  // Copy and sort history
  float sorted[FILTER_SIZE];
  for (int i = 0; i < FILTER_SIZE; i++) {
    sorted[i] = distanceHistory[i];
  }
  
  // Bubble sort
  for (int i = 0; i < FILTER_SIZE - 1; i++) {
    for (int j = 0; j < FILTER_SIZE - i - 1; j++) {
      if (sorted[j] > sorted[j+1]) {
        float temp = sorted[j];
        sorted[j] = sorted[j+1];
        sorted[j+1] = temp;
      }
    }
  }
  
  // Return median
  return sorted[FILTER_SIZE / 2];
}

/**
 * Encodes the current sonar scan telemetry packet as JSON and writes to Serial.
 */
void sendTelemetry(int angle, float distance, unsigned long pulseUs) {
  // We use standard Serial.print to construct the JSON string dynamically,
  // avoiding standard heavy C++ String object allocations on the ESP32 heap.
  Serial.print("{\"angle\":");
  Serial.print(angle);
  Serial.print(",\"distance\":");
  if (distance < 0) {
    Serial.print("null");
  } else {
    Serial.print(distance, 3);
  }
  Serial.print(",\"pulse_us\":");
  Serial.print(pulseUs);
  Serial.print(",\"timestamp\":");
  Serial.print(millis());
  Serial.println("}");
}
