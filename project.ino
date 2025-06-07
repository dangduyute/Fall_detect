#include "model.h"
#include <Chirale_TensorFlowLite.h>
#include <Wire.h>
#include <MPU6050.h>
#include <Adafruit_MLX90614.h>
#include <MAX30100_PulseOximeter.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"

// Firebase
#include <WiFi.h>
#include <FirebaseESP32.h>

// WiFi + Firebase cấu hình
#define WIFI_SSID "Viettel_6 Thanh"
#define WIFI_PASSWORD "0915650020"
#define FIREBASE_AUTH "fQ4pv2YhdPdEcnMHKYgdRXxVgY9AeZC9G57sLcm8"
#define FIREBASE_HOST "https://healthtakecare-edff9-default-rtdb.firebaseio.com/"

// Khởi tạo Firebase
FirebaseData firebaseData;
FirebaseAuth auth;
FirebaseConfig config;
String path = "/";

// Khai báo cảm biến
MPU6050 mpu;
Adafruit_MLX90614 mlx = Adafruit_MLX90614();
PulseOximeter pox;

// TensorFlow Lite
#define SEQ_LEN 400
#define N_FEATURES 6
const tflite::Model* model = tflite::GetModel(model_data);
tflite::AllOpsResolver resolver;
constexpr int kTensorArenaSize = 110 * 1024;
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
float (*sequence)[N_FEATURES] = nullptr;

const char* LABELS[] = { "fall", "lfall", "light", "rfall", "sit", "step", "walk" };

// Biến lưu cảm biến
float ambientTemp = 0;
float objectTemp = 0;
float bpm = 0;
float spo2 = 0;

// Cập nhật theo thời gian
unsigned long lastUpdateTemp = 0;
const unsigned long tempInterval = 10000; // 10 giây
bool maxUpdatedMidCycle = false;
// THÊM NGƯỠNG PHÁT HIỆN CHUYỂN ĐỘNG
const float acc_threshold = 12.0;   // m/s²
const float gyro_threshold = 150.0; // °/s

// Biến điều khiển trạng thái
bool isCollecting = false;
unsigned long lastMotionCheck = 0;
const unsigned long motionCheckInterval = 1000;

void setup() {
  Serial.begin(115200);
  Serial.println("Khởi động hệ thống...");

 
  Wire.begin(9, 8);
  Serial.println("📡 I2C Bus 0 initialized (SDA=9, SCL=8)");
   if (!pox.begin()) {
    Serial.println("Không thể khởi động MAX30100!");
    while (1) delay(1000);
  }
  Serial.println("MAX30100 connected");
  pox.setIRLedCurrent(MAX30100_LED_CURR_7_6MA);
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("Không thể kết nối MPU6050!");
    while (1) delay(1000);
  }
  Serial.println("MPU6050 connected");
  
  // Khởi tạo MAX30100
 

  
  Wire1.begin(6, 7);
  Serial.println("📡 I2C Bus 1 initialized (SDA=6, SCL=7)");
  
  if (!mlx.begin(0x5A, &Wire1)) {
    Serial.println("Không thể khởi động MLX90614 trên Wire1!");
    while (1) delay(1000);
  }
  Serial.println("MLX90614 connected");

  // Firebase config
  config.host = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;

  // Kết nối Firebase
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
  if (tensor_arena == nullptr) {
    Serial.println("Không thể cấp phát tensor_arena!");
    while (1) delay(1000);
  }
  

  sequence = (float(*)[N_FEATURES])heap_caps_malloc(sizeof(float) * SEQ_LEN * N_FEATURES, MALLOC_CAP_SPIRAM);
  if (sequence == nullptr) {
    Serial.println("Không thể cấp phát sequence buffer!");
    while (1) delay(1000);
  }
  
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Không thể cấp phát tensors!");
    while (1) delay(1000);
  }
  
  input = interpreter->input(0);
  output = interpreter->output(0);
  
}

void runInference() {
  Serial.println("Bắt đầu thu thập dữ liệu trong 20s (20Hz)...");

  for (int i = 0; i < SEQ_LEN; i++) {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    sequence[i][0] = (ax / 16384.0f) * 9.80665f;
    sequence[i][1] = (ay / 16384.0f) * 9.80665f;
    sequence[i][2] = (az / 16384.0f) * 9.80665f;
    sequence[i][3] = gx / 131.0f;
    sequence[i][4] = gy / 131.0f;
    sequence[i][5] = gz / 131.0f;


    delay(50); // 20Hz
  }


  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < N_FEATURES; j++) {
      input->data.f[i * N_FEATURES + j] = sequence[i][j];
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Lỗi khi chạy mô hình!");
    return;
  }

  int predicted_class = -1;
  float max_score = -1.0f;

  Serial.println(" Kết quả dự đoán:");
  for (int i = 0; i < 7; i++) {
    float score = output->data.f[i];
    Serial.printf("   %s: %.3f\n", LABELS[i], score);
    if (score > max_score) {
      max_score = score;
      predicted_class = i;
    }
  }

  Serial.printf("Hành động nhận diện: %s (%.2f)\n", LABELS[predicted_class], max_score);

  if ((predicted_class <= 3) && max_score > 0.7f) {
    Serial.println("CẢNH BÁO: TÉ NGÃ PHÁT HIỆN!");
    Firebase.setString(firebaseData, "/alert/fall_detected", LABELS[predicted_class]);
    Firebase.setFloat(firebaseData, "/alert/confidence", max_score);
    Firebase.setInt(firebaseData, "/alert/timestamp", millis());
    Serial.println("Đã gửi cảnh báo lên Firebase!");
  } else {
    Serial.println("Không phát hiện té ngã.");
  }

}

void loop() {
  pox.update();
  unsigned long now = millis();

 
  if (!maxUpdatedMidCycle && (now - lastUpdateTemp > tempInterval / 2)) {
    bpm = pox.getHeartRate();
    spo2 = pox.getSpO2();
    Firebase.setFloat(firebaseData, "/sensor/bpm", bpm);
    Firebase.setFloat(firebaseData, "/sensor/spo2", spo2);
    Serial.printf("[MAX30100 - 5s] BPM: %.1f, SpO2: %.1f%%\n", bpm, spo2);

    maxUpdatedMidCycle = true;
  }


  if (now - lastUpdateTemp > tempInterval) {
    float ambient = mlx.readAmbientTempC();
    float object = mlx.readObjectTempC();

    if (!isnan(ambient) && !isnan(object)) {
      ambientTemp = ambient;
      objectTemp = object;

      Firebase.setFloat(firebaseData, "/sensor/ambient_temp", ambientTemp);
      Firebase.setFloat(firebaseData, "/sensor/object_temp", objectTemp);

      Serial.printf("[MLX90614 - 10s] Môi trường: %.1f°C, Cơ thể: %.1f°C\n", ambientTemp, objectTemp);
    }

    lastUpdateTemp = now;
    maxUpdatedMidCycle = false;
  }

  
  if (!isCollecting && now - lastMotionCheck > motionCheckInterval) {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    float accX = (ax / 16384.0f) * 9.80665f;
    float accY = (ay / 16384.0f) * 9.80665f;
    float accZ = (az / 16384.0f) * 9.80665f;
    float gyroX = gx / 131.0f;
    float gyroY = gy / 131.0f;
    float gyroZ = gz / 131.0f;

    if (abs(accX) > acc_threshold || abs(accY) > acc_threshold || abs(accZ) > acc_threshold ||
        abs(gyroX) > gyro_threshold || abs(gyroY) > gyro_threshold || abs(gyroZ) > gyro_threshold) {
      Serial.println("Phát hiện chuyển động mạnh - bắt đầu thu thập dữ liệu!");
      isCollecting = true;
    }

    lastMotionCheck = now;
  }


  if (isCollecting) {
    runInference();
    isCollecting = false;
  }

  delay(10); // tránh watchdog reset
}