#version 330 core
in vec2 UV;
out vec4 FragColor;

uniform sampler2D texture1;
// 3x3 affine transform applied to UV (vec3(UV,1.0))
uniform mat3 uTransform;

void main() {
    vec3 t = uTransform * vec3(UV, 1.0);
    vec2 uv = t.xy;
    // If the transformed UV lies outside the texture, discard the
    // fragment so the background shows through instead of clamping to
    // the edge texel (which produces the 'smeared' artifact when zooming).
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        discard;
    }
    FragColor = texture(texture1, uv);
}
