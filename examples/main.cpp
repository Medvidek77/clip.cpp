// main example to demonstrate usage of the API

#include "clip.h"
#include "common-clip.h"
#include <vector>

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

    // To ensure both text and vision work without assertion crash when the model has none
    const int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;
    std::vector<float> txt_vec(vec_dim);
    clip_text_encode(ctx, params.n_threads, &tokens, txt_vec.data(), false);

    for (size_t i = 0; i < params.image_paths.size(); i++) {
        const char * img_path = params.image_paths[i].c_str();
        clip_image_u8 img0;
        if (!clip_image_load_from_file(img_path, &img0)) {
            fprintf(stderr, "failed to load image from '%s'\n", img_path);
            continue;
        }

        clip_image_f32 img_res;
        clip_image_preprocess(ctx, &img0, &img_res);

        std::vector<float> img_vec(vec_dim);
        clip_image_encode(ctx, params.n_threads, &img_res, img_vec.data(), false);

        float score = clip_similarity_score(img_vec.data(), txt_vec.data(), vec_dim);
        printf("[%2.3f] %s\n", score, img_path);
    }

    const int64_t t_main_end_us = ggml_time_us();

    if (params.verbose >= 1) {
        printf("\n\nTimings\n");
        printf("%s: Total time: %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);
    }

    clip_free(ctx);

    return 0;
}
