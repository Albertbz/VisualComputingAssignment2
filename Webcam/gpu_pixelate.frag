// language: glsl
#version 330 core
in vec2 UV;
out vec4 FragColor;

uniform sampler2D texture1; // your video texture

void main() {
    // size of one block in UV space
    ivec2 sz = textureSize(texture1, 0);
    vec2 texSize = vec2(sz);
    vec2 blockUV = 10 / texSize;

    vec2 blockOrigin = floor(UV / blockUV) * blockUV;
    vec2 center = blockOrigin + 0.5 * blockUV;

    vec2 q = blockUV * 0.25; // quarter-block offsets
    vec4 c1 = texture(texture1, center + vec2(-q.x, -q.y));
    vec4 c2 = texture(texture1, center + vec2( q.x, -q.y));
    vec4 c3 = texture(texture1, center + vec2(-q.x,  q.y));
    vec4 c4 = texture(texture1, center + vec2( q.x,  q.y));
    FragColor = (c1 + c2 + c3 + c4) * 0.25;
}