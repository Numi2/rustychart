// Simple WebGL2 fragment shader for candles.
#version 300 es
precision highp float;

in vec4 v_color;
out vec4 out_color;

void main() {
    out_color = v_color;
}

in vec4 v_color;
out vec4 o_color;

void main() {
    o_color = v_color;
}
