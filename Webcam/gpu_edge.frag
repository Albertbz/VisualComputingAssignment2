#version 330 core
in vec2 UV;
out vec4 FragColor;

uniform sampler2D texture1;

// texelOffset can be set from the program if desired. If not provided,
// the shader uses a reasonable default.
uniform vec2 texelOffset; // expected to be (1.0/width, 1.0/height)

void main() {
    vec2 off = texelOffset;
    if (off.x == 0.0 && off.y == 0.0) {
        off = vec2(1.0/512.0, 1.0/512.0);
    }

    // Simple Sobel-like kernel approximation using luminance
    float tl = dot(texture(texture1, UV + vec2(-off.x, -off.y)).rgb, vec3(0.299,0.587,0.114));
    float tc = dot(texture(texture1, UV + vec2(0.0, -off.y)).rgb, vec3(0.299,0.587,0.114));
    float tr = dot(texture(texture1, UV + vec2(off.x, -off.y)).rgb, vec3(0.299,0.587,0.114));

    float ml = dot(texture(texture1, UV + vec2(-off.x, 0.0)).rgb, vec3(0.299,0.587,0.114));
    float mr = dot(texture(texture1, UV + vec2(off.x, 0.0)).rgb, vec3(0.299,0.587,0.114));

    float bl = dot(texture(texture1, UV + vec2(-off.x, off.y)).rgb, vec3(0.299,0.587,0.114));
    float bc = dot(texture(texture1, UV + vec2(0.0, off.y)).rgb, vec3(0.299,0.587,0.114));
    float br = dot(texture(texture1, UV + vec2(off.x, off.y)).rgb, vec3(0.299,0.587,0.114));

    float gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);
    float gy = (bl + 2.0*bc + br) - (tl + 2.0*tc + tr);

    float g = length(vec2(gx, gy));
    FragColor = vec4(vec3(g), 1.0);
}
