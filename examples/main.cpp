// main example to demonstrate usage of the API

#include "clip.h"
#include "common-clip.h"
#include <vector>
#include <cmath>
#include <iostream>

void extract_tiles(const clip_image_u8* img, int target_size, std::vector<clip_image_u8>& tiles) {
    // Global context
    clip_image_u8 global_img;
    global_img.nx = img->nx;
    global_img.ny = img->ny;
    global_img.size = img->size;
    global_img.data = new uint8_t[img->size];
    std::copy(img->data, img->data + img->size, global_img.data);
    tiles.push_back(global_img);

    // If image is small enough, no tiling needed
    if (img->nx <= target_size && img->ny <= target_size) {
        return;
    }

    // Optimized approach: resize shortest side to target_size, then take up to 3 crops along the longest side.
    float scale = (float)target_size / std::min(img->nx, img->ny);
    int new_nx = std::max(target_size, (int)std::round(img->nx * scale));
    int new_ny = std::max(target_size, (int)std::round(img->ny * scale));

    // Very simple and fast nearest neighbor downscaling
    std::vector<uint8_t> scaled_data(3 * new_nx * new_ny);
    for (int y = 0; y < new_ny; y++) {
        int src_y = std::min((int)(y / scale), img->ny - 1);
        for (int x = 0; x < new_nx; x++) {
            int src_x = std::min((int)(x / scale), img->nx - 1);
            int src_idx = 3 * (src_y * img->nx + src_x);
            int dst_idx = 3 * (y * new_nx + x);

            scaled_data[dst_idx] = img->data[src_idx];
            scaled_data[dst_idx + 1] = img->data[src_idx + 1];
            scaled_data[dst_idx + 2] = img->data[src_idx + 2];
        }
    }

    // Extract up to 3 overlapping tiles from the resized image (start, center, end)
    std::vector<std::pair<int, int>> offsets;

    if (new_nx > new_ny) {
        // Horizontal image
        offsets.push_back({0, 0});
        if (new_nx > target_size) {
            offsets.push_back({(new_nx - target_size) / 2, 0});
            offsets.push_back({new_nx - target_size, 0});
        }
    } else {
        // Vertical or square image
        offsets.push_back({0, 0});
        if (new_ny > target_size) {
            offsets.push_back({0, (new_ny - target_size) / 2});
            offsets.push_back({0, new_ny - target_size});
        }
    }

    for (size_t i = 0; i < offsets.size(); i++) {
        // Check for duplicates
        bool is_duplicate = false;
        for (size_t j = 0; j < i; j++) {
            if (offsets[i] == offsets[j]) {
                is_duplicate = true;
                break;
            }
        }
        if (is_duplicate) continue;

        int offset_x = offsets[i].first;
        int offset_y = offsets[i].second;

        clip_image_u8 tile;
        tile.nx = target_size;
        tile.ny = target_size;
        tile.size = 3 * target_size * target_size;
        tile.data = new uint8_t[tile.size];

        for (int ty = 0; ty < target_size; ty++) {
            for (int tx = 0; tx < target_size; tx++) {
                int src_idx = 3 * ((offset_y + ty) * new_nx + (offset_x + tx));
                int dst_idx = 3 * (ty * target_size + tx);
                tile.data[dst_idx] = scaled_data[src_idx];
                tile.data[dst_idx + 1] = scaled_data[src_idx + 1];
                tile.data[dst_idx + 2] = scaled_data[src_idx + 2];
            }
        }
        tiles.push_back(tile);
    }
}

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    app_params params;
    if (!app_params_parse(argc, argv, params, 1, 1)) {
        print_help(argc, argv, params, 1, 1);
        return 1;
    }

    const int64_t t_load_us = ggml_time_us();

    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx) {
        printf("%s: Unable  to load model from %s", __func__, params.model.c_str());
        return 1;
    }

    if (params.image_paths.empty()) {
        fprintf(stderr, "%s: no image paths specified\n", __func__);
        return 1;
    }

    const char * text = params.texts[0].c_str();

    // Encode text only once
    clip_tokens tokens;
    clip_tokenize(ctx, text, &tokens);

    // Check if the model is a two-tower model
    const int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;
    std::vector<float> txt_vec(vec_dim);
    clip_text_encode(ctx, params.n_threads, &tokens, txt_vec.data(), true);

    const int target_size = clip_get_vision_hparams(ctx)->image_size;

    for (size_t i = 0; i < params.image_paths.size(); i++) {
        const char * img_path = params.image_paths[i].c_str();
        clip_image_u8 img0;
        if (!clip_image_load_from_file(img_path, &img0)) {
            fprintf(stderr, "failed to load image from '%s'\n", img_path);
            continue;
        }

        std::vector<clip_image_u8> tiles;
        extract_tiles(&img0, target_size, tiles);

        std::vector<float> avg_vec(vec_dim, 0.0f);

        for (const auto& tile : tiles) {
            clip_image_f32 img_res;
            clip_image_preprocess(ctx, &tile, &img_res);

            std::vector<float> img_vec(vec_dim);
            clip_image_encode(ctx, params.n_threads, &img_res, img_vec.data(), true);

            for (int j = 0; j < vec_dim; j++) {
                avg_vec[j] += img_vec[j];
            }

            delete[] img_res.data;
        }

        // Normalize the averaged vector
        float length = 0.0f;
        for (int j = 0; j < vec_dim; j++) {
            length += avg_vec[j] * avg_vec[j];
        }
        length = std::sqrt(length);
        for (int j = 0; j < vec_dim; j++) {
            avg_vec[j] /= length;
        }

        float score = clip_similarity_score(avg_vec.data(), txt_vec.data(), vec_dim);
        printf("[%2.3f] %s\n", score, img_path);

        // Free tile memory
        for (auto& tile : tiles) {
            delete[] tile.data;
        }
    }

    const int64_t t_main_end_us = ggml_time_us();

    if (params.verbose >= 1) {
        printf("\n\nTimings\n");
        printf("%s: Total time: %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);
    }

    clip_free(ctx);

    return 0;
}
