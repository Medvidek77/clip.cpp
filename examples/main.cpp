// main example to demonstrate usage of the API

#include "clip.h"
#include "common-clip.h"
#include <vector>
#include <cmath>
#include <iostream>

void extract_tiles(const clip_image_u8* img, int target_size, std::vector<clip_image_u8>& tiles) {
    // Strategy: calculate aspect ratio. If both dimensions are close to target_size, just return the image.
    // Otherwise, generate overlapping tiles of size target_size x target_size.
    // Also include a downscaled version of the whole image (which clip_image_preprocess does by default).

    // First, always include the original image as the global context
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

    // Determine number of tiles
    int nx_tiles = std::max(1, (int)std::ceil((float)img->nx / target_size));
    int ny_tiles = std::max(1, (int)std::ceil((float)img->ny / target_size));

    // For single tile dimensions, don't re-add what is already covered by the global context if it's perfectly sized
    if (nx_tiles == 1 && ny_tiles == 1) return;

    // Calculate strides to cover the image
    int stride_x = nx_tiles > 1 ? (img->nx - target_size) / (nx_tiles - 1) : 0;
    int stride_y = ny_tiles > 1 ? (img->ny - target_size) / (ny_tiles - 1) : 0;

    for (int y = 0; y < ny_tiles; y++) {
        for (int x = 0; x < nx_tiles; x++) {
            int offset_x = x * stride_x;
            int offset_y = y * stride_y;

            clip_image_u8 tile;
            tile.nx = target_size;
            tile.ny = target_size;
            tile.size = 3 * target_size * target_size;
            tile.data = new uint8_t[tile.size];

            for (int ty = 0; ty < target_size; ty++) {
                for (int tx = 0; tx < target_size; tx++) {
                    int src_idx = 3 * ((offset_y + ty) * img->nx + (offset_x + tx));
                    int dst_idx = 3 * (ty * target_size + tx);
                    tile.data[dst_idx] = img->data[src_idx];
                    tile.data[dst_idx + 1] = img->data[src_idx + 1];
                    tile.data[dst_idx + 2] = img->data[src_idx + 2];
                }
            }
            tiles.push_back(tile);
        }
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
