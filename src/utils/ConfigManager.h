// ConfigManager.h (Basitleştirilmiş Versiyon)
#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>

class ConfigManager {
public:
	// Yapıcı metot, dosya yolunu alır ve dosyayı yükleyip doğrular.
	// Hata durumunda std::runtime_error fırlatır.
	explicit ConfigManager(const std::filesystem::path& configPath);
	~ConfigManager() = default;

	// Ayarları almak için genel bir arayüz
	std::string getVideoPath() const;
	std::string getYoloModelPath() const;
	std::string getSAM2ModelPath() const;
	std::string getOutputPath() const;
	bool shouldShowKeypoints() const;
	bool shouldShowBoundingBoxes() const;
	float getConfidenceThreshold() const;
	float getNMSThreshold() const;
	int getMaxDetections() const;
	float getMaskAlpha() const;
	int getKeypointRadius() const;
	int getLineThickness() const;

	// Ayarları ekrana yazdırmak için
	void printConfig() const;

private:
	nlohmann::json config;
	std::filesystem::path configFilePath;

	// Değerleri JSON'dan güvenli bir şekilde almak için özel yardımcı fonksiyon (template)
	template<typename T>
	T getValue(const std::string& section, const std::string& key, const T& defaultValue) const;

	// Doğrulama fonksiyonları
	void validateConfig();
};