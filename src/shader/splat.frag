#version 460
#extension GL_EXT_shader_atomic_float : enable

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 position;

layout(location = 0) out vec4 out_color;
layout(push_constant) uniform constants {
    mat4 model;
    float alpha_threshold;
    float splat_scale;
} PushConstants;
//layout(set = 2, binding = 0, r32ui) uniform uimage2D counter_image;

void main() {

  ivec2 pixel = ivec2(gl_FragCoord.xy);

  // Atomically increment the counter for this pixel
  //uint count = imageAtomicAdd(counter_image, pixel, 1);

  
  // uint max_splats = 4;
  // if (count >= max_splats)
  // {
  //     discard;
  // }
  float alpha_threshold = PushConstants.alpha_threshold;
  
  float gaussian_alpha = exp(-0.5f * dot(position, position));
  float alpha = color.a * gaussian_alpha;
  if (alpha < alpha_threshold) {
      discard;
  }
  out_color = vec4(color.rgb, alpha);
}
