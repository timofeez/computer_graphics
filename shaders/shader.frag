#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

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

struct DirectionalLight {
	vec3 direction;
	float _padding1;
	vec3 color;
	float intensity;
};

struct PointLight {
	vec3 position;
	float _padding1;
	vec3 color;
	float intensity;
};

layout (binding = 2, std140) uniform LightingUniformsBlock {
	DirectionalLight directional;
	vec3 ambient_color;
	float ambient_intensity;
	uint point_light_count;
	float _padding2;
	float _padding3;
	float _padding4;
} lighting_uniforms;

layout (binding = 3, std430) readonly buffer PointLightsBuffer {
	PointLight lights[];
} point_lights;

layout (binding = 4) uniform sampler2D texture_sampler;

vec3 blinnPhong(vec3 normal, vec3 view_dir, vec3 light_dir, vec3 light_color, float light_intensity) {
	normal = normalize(normal);
	view_dir = normalize(view_dir);
	light_dir = normalize(light_dir);
	
	// Diffuse component
	float ndotl = max(dot(normal, light_dir), 0.0);
	vec3 diffuse = model_data.albedo_color * light_color * light_intensity * ndotl;
	
	// Specular component (Blinn-Phong)
	vec3 half_dir = normalize(light_dir + view_dir);
	float ndoth = max(dot(normal, half_dir), 0.0);
	vec3 specular = model_data.specular_color * light_color * light_intensity * pow(ndoth, model_data.shininess);
	
	return diffuse + specular;
}

void main() {
	vec3 normal = normalize(f_normal);
	vec3 view_dir = normalize(scene.camera_position - f_position);
	
	// Sample texture using texture coordinates
	vec4 texture_color = texture(texture_sampler, f_uv);
	
	// Ambient lighting
	vec3 ambient = texture_color.rgb * lighting_uniforms.ambient_color * lighting_uniforms.ambient_intensity;
	
	// Directional light
	vec3 dir_light_dir = normalize(-lighting_uniforms.directional.direction);
	vec3 dir_light = blinnPhong(normal, view_dir, dir_light_dir, lighting_uniforms.directional.color, lighting_uniforms.directional.intensity);
	dir_light *= texture_color.rgb; // Modulate with texture color
	
	// Point lights with inverse square law attenuation
	vec3 point_light_contribution = vec3(0.0);
	for (uint i = 0; i < min(lighting_uniforms.point_light_count, 32u); ++i) {
		PointLight light = point_lights.lights[i];
		vec3 light_dir = light.position - f_position;
		float distance = length(light_dir);
		
		// Inverse square law attenuation
		float attenuation = 1.0 / (distance * distance);
		
		vec3 point_light = blinnPhong(normal, view_dir, light_dir, light.color, light.intensity * attenuation);
		point_light *= texture_color.rgb; // Modulate with texture color
		point_light_contribution += point_light;
	}
	
	vec3 final = ambient + dir_light + point_light_contribution;
	final_color = vec4(final, texture_color.a);
}
