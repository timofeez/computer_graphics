#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_position_light_space;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
	float _padding1;
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	float use_texture; // 1.0 to use texture, 0.0 to use albedo_color only
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
layout (binding = 5) uniform sampler2D shadow_map; // Depth texture - manual comparison

float calculateShadow(vec4 frag_pos_light_space, vec3 normal, vec3 light_dir) {
	// Perform perspective divide to get NDC coordinates
	vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
	
	// Transform from NDC [-1,1] to texture coordinates [0,1]
	vec2 tex_coords = proj_coords.xy * 0.5 + 0.5;
	float current_depth = proj_coords.z;
	
	// Check if fragment is outside light's view frustum
	if (tex_coords.x < 0.0 || tex_coords.x > 1.0 ||
	    tex_coords.y < 0.0 || tex_coords.y > 1.0 ||
	    current_depth < 0.0 || current_depth > 1.0) {
		return 1.0; // Outside frustum = lit
	}
	
	// Clamp texture coordinates to avoid sampling outside texture
	tex_coords = clamp(tex_coords, 0.001, 0.999);
	current_depth = clamp(current_depth, 0.0, 1.0);
	
	// Sample depth from shadow map
	// For D32_SFLOAT format read as sampler2D, depth is stored in red channel
	float shadow_map_depth = texture(shadow_map, tex_coords).r;
	
	// Add depth bias to prevent self-shadowing (shadow acne)
	// Adaptive bias based on surface angle
	float ndotl = max(dot(normal, light_dir), 0.0);
	float bias = max(0.005 * (1.0 - ndotl), 0.0005);
	float biased_depth = max(0.0, current_depth - bias);
	
	// Depth comparison: 
	// - If biased_depth > shadow_map_depth: fragment is behind shadow caster = IN SHADOW (return 0.0)
	// - If biased_depth <= shadow_map_depth: fragment is in front = LIT (return 1.0)
	// For Vulkan: smaller depth values = closer to camera/light (near plane = 0.0)
	// float shadow = step(biased_depth, shadow_map_depth); // Returns 1.0 if biased_depth <= shadow_map_depth
	
	// Apply PCF (Percentage Closer Filtering) for smoother shadow edges
	vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
	float shadow_sum = 0.0;
	float sample_count = 0.0;
	
	// Sample 3x3 neighborhood
	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			// if (x == 0 && y == 0) continue; // Skip center sample (already added)
			
			// Widen the sampling kernel slightly for softer edges
			vec2 offset = vec2(float(x), float(y)) * texel_size * 1.5;
			vec2 sample_uv = clamp(tex_coords + offset, 0.001, 0.999);
			float sample_depth = texture(shadow_map, sample_uv).r;
			
			// Compare depth at this sample location
			shadow_sum += step(biased_depth, sample_depth);
			sample_count += 1.0;
		}
	}
	
	// Average shadow across all samples
	return shadow_sum / sample_count;
}

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
	
	vec4 texture_color = texture(texture_sampler, f_uv);
	vec3 base_color = mix(model_data.albedo_color, texture_color.rgb, model_data.use_texture);
	float base_alpha = mix(1.0, texture_color.a, model_data.use_texture);
	
	vec3 ambient = base_color * lighting_uniforms.ambient_color * lighting_uniforms.ambient_intensity;
	
	vec3 dir_light_dir = normalize(-lighting_uniforms.directional.direction);
	float shadow = calculateShadow(f_position_light_space, normal, dir_light_dir);
	
	vec3 dir_light = blinnPhong(normal, view_dir, dir_light_dir, lighting_uniforms.directional.color, lighting_uniforms.directional.intensity);
	dir_light *= base_color;
	
	// Make shadows very dark (almost black) when in shadow
	// Soften shadow based on distance? No, shadow map doesn't support variable penumbra easily.
	// But we can fake it or just ensure the edges are soft with PCF.
	// The current PCF is doing a good job.
	dir_light *= mix(0.3, 1.0, shadow);
	
	vec3 point_light_contribution = vec3(0.0);
	for (uint i = 0; i < min(lighting_uniforms.point_light_count, 32u); ++i) {
		PointLight light = point_lights.lights[i];
		vec3 light_dir = light.position - f_position;
		float distance = length(light_dir);
		float attenuation = 1.0 / (distance * distance);
		vec3 point_light = blinnPhong(normal, view_dir, light_dir, light.color, light.intensity * attenuation);
		point_light *= base_color;
		point_light_contribution += point_light;
	}
	
	vec3 final = ambient + dir_light + point_light_contribution;
	final_color = vec4(final, base_alpha);

	// DEBUG: Visualize Shadow Map
	// Use passed light space position instead of recalculating
	vec3 proj_coords = f_position_light_space.xyz / f_position_light_space.w;
	vec2 tex_coords = proj_coords.xy * 0.5 + 0.5;
	
	// Check if inside frustum
	if (tex_coords.x >= 0.0 && tex_coords.x <= 1.0 &&
	    tex_coords.y >= 0.0 && tex_coords.y <= 1.0) {
		// float depth_sample = texture(shadow_map, tex_coords).r;
		// Display depth map on the object
		// final_color = vec4(vec3(depth_sample), 1.0);
	}
	
	// DEBUG VISUALIZATION - UNCOMMENT TO SEE SHADOW MAP
	// final_color = vec4(vec3(shadow), 1.0);
}
