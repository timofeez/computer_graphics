#version 450

layout (location = 0) in vec3 v_position;

layout (binding = 0, std140) uniform LightUniforms {
	mat4 light_view_projection;
} light;

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	float use_texture; // 1.0 to use texture, 0.0 to use albedo_color only
	vec3 specular_color;
	float shininess;
} model_data;

void main() {
	vec4 position = model_data.model * vec4(v_position, 1.0f);
	gl_Position = light.light_view_projection * position;
}

