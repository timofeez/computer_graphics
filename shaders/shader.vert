#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
	float _padding1;
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	float _padding1;
	vec3 specular_color;
	float shininess;
} model_data;

void main() {
	vec4 position = model_data.model * vec4(v_position, 1.0f);
	
	// Transform normal using inverse transpose of model matrix for proper normal transformation
	mat3 normal_matrix = mat3(transpose(inverse(model_data.model)));
	vec3 normal = normalize(normal_matrix * v_normal);

	gl_Position = scene.view_projection * position;

	f_position = position.xyz;
	f_normal = normal;
	f_uv = v_uv;
}
