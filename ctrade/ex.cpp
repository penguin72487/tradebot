#include <iostream>
#include <curl/curl.h>
#include <string>

// Callback function to receive data from libcurl
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main() {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://example.com"); // URL to fetch
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback); // Set callback
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer); // Pass the string buffer to the callback function

        res = curl_easy_perform(curl); // Perform the request

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Fetched Data: \n" << readBuffer << std::endl; // Print the fetched data
        }

        curl_easy_cleanup(curl); // Clean up
    }

    curl_global_cleanup();

    return 0;
}
