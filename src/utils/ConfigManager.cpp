// ConfigManager.cpp (Basitleştirilmiş Versiyon)
#include "ConfigManager.h"
#include <fstream>
#include <iostream>

// Yapıcı metot: Dosyayı yükler ve doğrular
ConfigManager::ConfigManager(const std::filesystem::path& path) : configFilePath(path) {
    if (!std::filesystem::exists(configFilePath)) {
        throw std::runtime_error("Yapılandırma dosyası bulunamadı: " + configFilePath.string());
    }

    std::ifstream file(configFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Yapılandırma dosyası açılamadı: " + configFilePath.string());
    }

    try {
        file >> config;
        validateConfig(); // Yükleme sonrası hemen doğrula
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parse hatası: " + std::string(e.what()));
    }
}

// Özel yardımcı template fonksiyonu
template<typename T>
T ConfigManager::getValue(const std::string& section, const std::string& key, const T& defaultValue) const {
    // config[section] var mı diye kontrol et, yoksa varsayılanı kullan
    if (config.contains(section) && config[section].contains(key)) {
        return config[section].value(key, defaultValue);
    }
    return defaultValue;
}

// Doğrulama
void ConfigManager::validateConfig() {
    if (getVideoPath().empty() || !std::filesystem::exists(getVideoPath())) {
        throw std::runtime_error("Geçersiz veya eksik video yolu: " + getVideoPath());
    }
    if (getYoloModelPath().empty() || !std::filesystem::exists(getYoloModelPath())) {
        throw std::runtime_error("Geçersiz veya eksik YOLO modeli yolu: " + getYoloModelPath());
    }
    if (getSAM2ModelPath().empty() || !std::filesystem::exists(getSAM2ModelPath())) {
        throw std::runtime_error("Geçersiz veya eksik SAM2 modeli yolu: " + getSAM2ModelPath());
    }
    // Diğer doğrulamalar (thresholds vs.) buraya eklenebilir.
}

// Arayüz fonksiyonları artık sadece `getValue`'ı çağırır
std::string ConfigManager::getVideoPath() const { return getValue<std::string>("input", "video_path", ""); }
std::string ConfigManager::getYoloModelPath() const { return getValue<std::string>("models", "yolov8_model_path", ""); }
std::string ConfigManager::getSAM2ModelPath() const { return getValue<std::string>("models", "sam2_model_path", ""); }
std::string ConfigManager::getOutputPath() const { return getValue<std::string>("output", "output_path", "./output/processed.mp4"); }
bool ConfigManager::shouldShowKeypoints() const { return getValue<bool>("output", "show_keypoints", true); }
bool ConfigManager::shouldShowBoundingBoxes() const { return getValue<bool>("output", "show_bounding_boxes", true); }
float ConfigManager::getConfidenceThreshold() const { return getValue<float>("processing", "confidence_threshold", 0.5f); }
float ConfigManager::getNMSThreshold() const { return getValue<float>("processing", "nms_threshold", 0.4f); }
int ConfigManager::getMaxDetections() const { return getValue<int>("processing", "max_detections", 100); }
float ConfigManager::getMaskAlpha() const { return getValue<float>("visualization", "mask_alpha", 0.6f); }
int ConfigManager::getKeypointRadius() const { return getValue<int>("visualization", "keypoint_radius", 3); }
int ConfigManager::getLineThickness() const { return getValue<int>("visualization", "line_thickness", 2); }

void ConfigManager::printConfig() const {
    std::cout << "=== Configuration Summary ===" << std::endl;
    std::cout << "Config File: " << configFilePath.string() << std::endl;
    std::cout << "Input Video: " << getVideoPath() << std::endl;
    std::cout << "YOLOv8 Model: " << getYoloModelPath() << std::endl;
    std::cout << "============================" << std::endl;
}
