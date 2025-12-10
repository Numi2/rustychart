// Simple WebGL2 vertex shader for candle geometry.
#version 300 es
precision highp float;

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;

out vec4 v_color;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_color = color;
}

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec4 a_color;

out vec4 v_color;

void main() {
    v_color = a_color;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
