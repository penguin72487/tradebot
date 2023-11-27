#include <iostream>
#include <curl/curl.h>

size_t callback(const char* in, size_t size, size_t num, std::string* out) {
    const size_t totalBytes(size * num);
    out->append(in, totalBytes);
    return totalBytes;
}

int main() {
    CURL *curl;
    CURLcode res;
    std::string response;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if(curl) {
        // 设置请求的 URL
        curl_easy_setopt(curl, CURLOPT_URL, "https://testnet.binance.vision/api/v3/time");

        // 设置回调函数，以便收集响应
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);

        // 设置接收数据的变量
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // 发送请求
        res = curl_easy_perform(curl);

        // 检查错误
        if(res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Response: " << response << std::endl;
        }

        // 清理
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
    return 0;
}
