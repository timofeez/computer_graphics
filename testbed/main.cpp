#include "veekay/input.hpp"
#include <format>
#include <climits>
#include <vector>
#include <stack>
#include <iostream>
#include <fstream>
#include <cmath>
#include <veekay/veekay.hpp>
#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 32;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::vec3 camera_position;
	float _padding1;
};

struct alignas(16) ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color;
	float _padding1;
	veekay::vec3 specular_color;
	float shininess;
};

struct alignas(16) DirectionalLight {
	veekay::vec3 direction;
	float _padding1;
	veekay::vec3 color;
	float intensity;
};

struct alignas(16) PointLight {
	veekay::vec3 position;
	float _padding1;
	veekay::vec3 color;
	float intensity;
};

struct alignas(16) LightingUniforms {
	DirectionalLight directional;
	veekay::vec3 ambient_color;
	float ambient_intensity;
	uint32_t point_light_count;
	float _padding2;
	float _padding3;
	float _padding4;
};

struct Material {
	veekay::vec3 albedo;
	veekay::vec3 specular;
	float shininess;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* edge_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
	uint32_t edge_indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};
	[[nodiscard]] veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	Material material;
float rotation_speed = 1.0f;
	veekay::vec3 initial_rotation = {};
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;
	veekay::vec3 position = {};
	veekay::vec3 rotation = {};
	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;
	[[nodiscard]] veekay::mat4 view() const;
	[[nodiscard]] veekay::mat4 view_projection(float aspect_ratio) const;
};

inline namespace {
	Camera camera{
		.position = {10.0f, 10.0f, 9.0f},
		.rotation = {54.0f, -139.0f, -26.0f}
	};
	std::vector<Model> models;
	float global_rotation_speed = 1.5f;
	bool wireframe_mode = false;
	
	// Lighting
	LightingUniforms lighting{
		.directional = {
			.direction = veekay::vec3::normalized({-0.5f, -1.0f, -0.3f}),
			.color = {1.0f, 1.0f, 1.0f},
			.intensity = 1.0f
		},
		.ambient_color = {1.0f, 1.0f, 1.0f},
		.ambient_intensity = 0.2f
	};
	
	std::vector<PointLight> point_lights;
	uint32_t point_light_count = 0;
	
	// Camera controls
	bool mouse_captured = false;
	float camera_speed = 5.0f;
	float mouse_sensitivity = 0.1f;
}

inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;
	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;
		VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;
	VkPipeline wireframe_pipeline;
	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
	veekay::graphics::Buffer* lighting_uniforms_buffer;
	veekay::graphics::Buffer* point_lights_buffer;
	Mesh cone_mesh;
	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;
}

float toRadians(float degrees) {
	return degrees * static_cast<float>(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	const auto scaling_mtx = veekay::mat4::scaling(scale);
	const auto rot_mtx_x = veekay::mat4::rotation({1., .0, .0}, rotation.x);
	const auto rot_mtx_y = veekay::mat4::rotation({.0, -1., .0}, rotation.y);
	const auto rot_mtx_z = veekay::mat4::rotation({.0, .0, 1.}, rotation.z);
	auto t = veekay::mat4::translation(position);
	// Порядок X * Y * Z: сначала наклон X (наклоняет ось), затем вращение Y, затем Z
	return scaling_mtx * rot_mtx_x * rot_mtx_y * rot_mtx_z * t;
}

veekay::mat4 Camera::view() const {
	const auto t = veekay::mat4::translation(-position);
	const auto rot_mtx_x = veekay::mat4::rotation({1., .0, .0}, toRadians(rotation.x));
	const auto rot_mtx_y = veekay::mat4::rotation({.0, -1., .0}, toRadians(rotation.y));
	const auto rot_mtx_z = veekay::mat4::rotation({.0, .0, 1.}, toRadians(rotation.z));
	return t * rot_mtx_x * rot_mtx_y * rot_mtx_z;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection_mtx = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	return view() * projection_mtx;
}

VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << path << "\n";
		return nullptr;
	}
	std::streampos file_size = file.tellg();
	if (file_size <= 0) {
		std::cerr << "Invalid shader file size: " << path << "\n";
		file.close();
		return nullptr;
	}
	size_t size = static_cast<size_t>(file_size);
	if (size % sizeof(uint32_t) != 0) {
		std::cerr << "Shader file size is not a multiple of 4: " << path << "\n";
		file.close();
		return nullptr;
	}
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();
	if (file.gcount() != static_cast<std::streamsize>(size)) {
		std::cerr << "Failed to read entire shader file: " << path << "\n";
		return nullptr;
	}
	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};
	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}
	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
	{
		vertex_shader_module = loadShaderModule("shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}
		fragment_shader_module = loadShaderModule("shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}
		VkPipelineShaderStageCreateInfo stage_infos[2];
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			}
		};
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = std::size(attributes),
			.pVertexAttributeDescriptions = attributes,
		};
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};
		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor,
		};
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,
			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};
		VkPipelineColorBlendAttachmentState wireframe_attachment{
			.blendEnable = VK_TRUE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
							  VK_COLOR_COMPONENT_G_BIT |
							  VK_COLOR_COMPONENT_B_BIT |
							  VK_COLOR_COMPONENT_A_BIT,
		};
		VkPipelineColorBlendStateCreateInfo wireframe_blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.attachmentCount = 1,
			.pAttachments = &wireframe_attachment,
		};
		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
				}
			};
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = std::size(pools),
				.pPoolSizes = pools,
			};
			if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};
			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = std::size(bindings),
				.pBindings = bindings,
			};
			if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}
		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};
			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};
		if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};
		if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
		VkPipelineInputAssemblyStateCreateInfo line_assembly_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		};
		VkPipelineRasterizationStateCreateInfo line_raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};
		VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};
		VkPipelineDynamicStateCreateInfo dynamic_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = 1,
			.pDynamicStates = dynamic_states,
		};
		VkGraphicsPipelineCreateInfo line_pipeline_info = info;
		line_pipeline_info.pInputAssemblyState = &line_assembly_info;
		line_pipeline_info.pRasterizationState = &line_raster_info;
		line_pipeline_info.pColorBlendState = &wireframe_blend_info;
		line_pipeline_info.pDynamicState = &dynamic_state_info;
		if (vkCreateGraphicsPipelines(device, nullptr, 1, &line_pipeline_info, nullptr, &wireframe_pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create wireframe pipeline\n";
			veekay::app.running = false;
			return;
		}
	}
	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * sizeof(ModelUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	lighting_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(LightingUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	point_lights_buffer = new veekay::graphics::Buffer(
		max_point_lights * sizeof(PointLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};
		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}
		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};
		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}
	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
				.buffer = lighting_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(LightingUniforms),
			},
			{
				.buffer = point_lights_buffer->buffer,
				.offset = 0,
				.range = max_point_lights * sizeof(PointLight),
			},
		};
		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],
			},
		};
		vkUpdateDescriptorSets(device, std::size(write_infos), write_infos, 0, nullptr);
	}
	{
		constexpr float radius = 1.0f;
		constexpr float height = 2.0f;
		constexpr int segments = 32;
		std::vector<Vertex> vertices;
		vertices.push_back(Vertex{
			.position = {0.0f, height, 0.0f},
			.normal = {0.0f, 1.0f, 0.0f},
			.uv = {0.5f, 0.0f}
		});
		for (int i = 0; i <= segments; ++i) {
			float angle = static_cast<float>(i) * 2.0f * static_cast<float>(M_PI) / static_cast<float>(segments);
			float x = radius * cosf(angle);
			float z = radius * sinf(angle);
			float len = sqrtf(radius * radius + height * height);
			veekay::vec3 normal = {x / len, radius / len, z / len};
			normal = veekay::vec3::normalized(normal);
			vertices.push_back(Vertex{
				.position = {x, 0.0f, z},
				.normal = normal,
				.uv = {static_cast<float>(i) / static_cast<float>(segments), 1.0f}
			});
		}
		uint32_t base_center = static_cast<uint32_t>(vertices.size());
		vertices.push_back(Vertex{
			.position = {0.0f, 0.0f, 0.0f},
			.normal = {0.0f, -1.0f, 0.0f},
			.uv = {0.5f, 0.5f}
		});
		std::vector<uint32_t> indices;
		for (int i = 0; i < segments; ++i) {
			indices.push_back(0);
			indices.push_back(i + 1);
			indices.push_back((i + 1) % segments + 1);
		}
		for (int i = 0; i < segments; ++i) {
			indices.push_back(base_center);
			indices.push_back(i + 1);
			indices.push_back((i + 1) % segments + 1);
		}
		std::vector<uint32_t> edge_indices;
		for (int i = 1; i <= segments; ++i) {
			edge_indices.push_back(0);
			edge_indices.push_back(i);
		}
		for (int i = 1; i <= segments; ++i) {
			edge_indices.push_back(i);
			edge_indices.push_back((i % segments) + 1);
		}
		cone_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		cone_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		cone_mesh.edge_buffer = new veekay::graphics::Buffer(
			edge_indices.size() * sizeof(uint32_t), edge_indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		cone_mesh.indices = static_cast<uint32_t>(indices.size());
		cone_mesh.edge_indices = static_cast<uint32_t>(edge_indices.size());
	}
	
	// Enable mouse capture by default for camera controls
	veekay::input::mouse::setCaptured(mouse_captured);
	
	// Initialize point lights
	point_lights.emplace_back(PointLight{
		.position = {2.0f, 3.0f, 2.0f},
		.color = {1.0f, 1.0f, 1.0f},
		.intensity = 10.0f
	});
	point_lights.emplace_back(PointLight{
		.position = {-2.0f, 3.0f, -2.0f},
		.color = {1.0f, 0.5f, 0.3f},
		.intensity = 8.0f
	});
	point_light_count = static_cast<uint32_t>(point_lights.size());
	
	// Arrange models in an ordered grid with different colors
	// Row 1
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {-3.0f, 0.0f, -3.0f},
			.scale = {1.0f, 1.0f, 1.0f},
			.rotation = {1.0f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {1.0f, 0.2f, 0.2f}, // Red
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 32.0f
		},
		.rotation_speed = 1.0f,
		.initial_rotation = {1.0f, 0.0f, 0.0f}
	});
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {0.0f, 0.0f, -3.0f},
			.scale = {1.0f, 1.0f, 1.0f},
			.rotation = {1.5f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {0.2f, 1.0f, 0.2f}, // Green
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 64.0f
		},
		.rotation_speed = 1.0f,
		.initial_rotation = {1.5f, 0.0f, 0.0f}
	});
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {3.0f, 0.0f, -3.0f},
			.scale = {1.0f, 1.0f, 1.0f},
			.rotation = {2.0f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {0.2f, 0.2f, 1.0f}, // Blue
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 128.0f
		},
		.rotation_speed = 1.0f,
		.initial_rotation = {2.0f, 0.0f, 0.0f}
	});
	// Row 2
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {-1.5f, 0.0f, 0.0f},
			.scale = {1.0f, 1.0f, 1.0f},
			.rotation = {1.2f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {1.0f, 1.0f, 0.2f}, // Yellow
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 32.0f
		},
		.rotation_speed = 1.0f,
		.initial_rotation = {1.2f, 0.0f, 0.0f}
	});
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {1.5f, 0.0f, 0.0f},
			.scale = {1.0f, 1.0f, 1.0f},
			.rotation = {1.8f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {1.0f, 0.2f, 1.0f}, // Magenta
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 64.0f
		},
		.rotation_speed = 1.0f,
		.initial_rotation = {1.8f, 0.0f, 0.0f}
	});
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;
	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;
	delete cone_mesh.index_buffer;
	delete cone_mesh.vertex_buffer;
	delete cone_mesh.edge_buffer;
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;
	delete lighting_uniforms_buffer;
	delete point_lights_buffer;
	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipeline(device, wireframe_pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

veekay::vec3 getForwardVector(const veekay::vec3& rotation) {
	float pitch = toRadians(rotation.x);
	float yaw = toRadians(rotation.y);
	return veekay::vec3{
		cosf(pitch) * sinf(yaw),
		-sinf(pitch),
		cosf(pitch) * cosf(yaw)
	};
}

veekay::vec3 getRightVector(const veekay::vec3& rotation) {
	float yaw = toRadians(rotation.y);
	return veekay::vec3{
		sinf(yaw - M_PI / 2.0f),
		0.0f,
		cosf(yaw - M_PI / 2.0f)
	};
}

veekay::vec3 getUpVector(const veekay::vec3& forward, const veekay::vec3& right) {
	return veekay::vec3::normalized(veekay::vec3::cross(forward, right));
}

void update(double time) {
	static double last_time = time;
	double delta_time = time - last_time;
	last_time = time;
	
	// Camera controls
	if (veekay::input::keyboard::isKeyPressed(veekay::input::keyboard::Key::escape)) {
		mouse_captured = !mouse_captured;
		veekay::input::mouse::setCaptured(mouse_captured);
	}
	
	if (mouse_captured) {
		// Mouse look
		veekay::vec2 mouse_delta = veekay::input::mouse::cursorDelta();
		camera.rotation.y -= mouse_delta.x * mouse_sensitivity;
		camera.rotation.x -= mouse_delta.y * mouse_sensitivity;
		
		// Clamp pitch
		if (camera.rotation.x > 89.0f) camera.rotation.x = 89.0f;
		if (camera.rotation.x < -89.0f) camera.rotation.x = -89.0f;
		
		// Keyboard movement
		veekay::vec3 forward = getForwardVector(camera.rotation);
		veekay::vec3 right = getRightVector(camera.rotation);
		veekay::vec3 up = getUpVector(forward, right);
		
		float speed = camera_speed * static_cast<float>(delta_time);
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) {
			camera.position = camera.position + forward * speed;
		}
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) {
			camera.position = camera.position - forward * speed;
		}
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) {
			camera.position = camera.position - right * speed;
		}
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) {
			camera.position = camera.position + right * speed;
		}
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::space)) {
			camera.position.y += speed;
		}
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::left_shift)) {
			camera.position.y -= speed;
		}
	}
	
	// UI
	ImGui::Begin("Controls");
	ImGui::Checkbox("Wireframe", &wireframe_mode);
	ImGui::SliderFloat("Rotation Speed", &global_rotation_speed, 0.0f, 10.0f);
	
	ImGui::Separator();
	ImGui::Text("Models: %zu", models.size());
	for (size_t i = 0; i < models.size() && i < 5; ++i) {
		ImGui::Text("Model %zu: pos(%.1f, %.1f, %.1f)", i, 
			models[i].transform.position.x,
			models[i].transform.position.y,
			models[i].transform.position.z);
	}
	ImGui::Separator();
	ImGui::Text("Camera");
	ImGui::InputFloat3("Position", &camera.position.x);
	ImGui::InputFloat3("Rotation", &camera.rotation.x);
	ImGui::SliderFloat("Camera Speed", &camera_speed, 0.1f, 20.0f);
	ImGui::SliderFloat("Mouse Sensitivity", &mouse_sensitivity, 0.01f, 1.0f);
	ImGui::Text("Controls:");
	ImGui::Text("  ESC - Toggle mouse capture");
	ImGui::Text("  When captured: WASD - move, Mouse - rotate");
	ImGui::Text("  RMB (hold) - Rotate camera");
	ImGui::Text("Mouse Captured: %s", mouse_captured ? "YES" : "NO");
	
	ImGui::Separator();
	ImGui::Text("Lighting");
	ImGui::ColorEdit3("Ambient Color", &lighting.ambient_color.x);
	ImGui::SliderFloat("Ambient Intensity", &lighting.ambient_intensity, 0.0f, 1.0f);
	
	ImGui::Separator();
	ImGui::Text("Directional Light");
	ImGui::InputFloat3("Direction", &lighting.directional.direction.x);
	// Normalize direction after potential UI change
	lighting.directional.direction = veekay::vec3::normalized(lighting.directional.direction);
	ImGui::ColorEdit3("Directional Color", &lighting.directional.color.x);
	ImGui::SliderFloat("Directional Intensity", &lighting.directional.intensity, 0.0f, 2.0f);
	
	ImGui::Separator();
	ImGui::Text("Point Lights");
	ImGui::Text("Count: %u", point_light_count);
	for (size_t i = 0; i < point_lights.size() && i < max_point_lights; ++i) {
		ImGui::PushID(static_cast<int>(i));
		if (ImGui::TreeNode(std::format("Point Light {}", i).c_str())) {
			ImGui::InputFloat3("Position", &point_lights[i].position.x);
			ImGui::ColorEdit3("Color", &point_lights[i].color.x);
			ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 20.0f);
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	
	ImGui::End();
	
	// Update model rotations - вращаются вокруг наклонной оси Y
	// Применяем небольшой наклон к оси вращения (5 градусов)
	const float rotation_tilt = 5.0f;
	for (auto& model : models) {
		// Наклон оси вращения вокруг X (применяется перед Y, чтобы наклонить ось)
		model.transform.rotation.x = model.initial_rotation.x + rotation_tilt;
		// Вращение вокруг наклонной оси Y
		model.transform.rotation.y = model.initial_rotation.y + model.rotation_speed * global_rotation_speed * time;
		model.transform.rotation.z = model.initial_rotation.z;
	}
	
	// Update uniforms - optimize by writing directly to mapped memory
	float aspect_ratio = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height);
	
	// Write scene uniforms directly
	SceneUniforms* scene_uniforms = static_cast<SceneUniforms*>(scene_uniforms_buffer->mapped_region);
	scene_uniforms->view_projection = camera.view_projection(aspect_ratio);
	scene_uniforms->camera_position = camera.position;
	
	// Write model uniforms directly
	ModelUniforms* model_uniforms = static_cast<ModelUniforms*>(model_uniforms_buffer->mapped_region);
	for (size_t i = 0; i < models.size(); ++i) {
		model_uniforms[i].model = models[i].transform.matrix();
		model_uniforms[i].albedo_color = models[i].material.albedo;
		model_uniforms[i].specular_color = models[i].material.specular;
		model_uniforms[i].shininess = models[i].material.shininess;
	}
	
	// Write lighting uniforms directly
	LightingUniforms* lighting_data = static_cast<LightingUniforms*>(lighting_uniforms_buffer->mapped_region);
	*lighting_data = lighting;
	lighting_data->point_light_count = point_light_count;
	
	// Update point lights buffer directly
	PointLight* lights_data = static_cast<PointLight*>(point_lights_buffer->mapped_region);
	for (size_t i = 0; i < point_lights.size() && i < max_point_lights; ++i) {
		lights_data[i] = point_lights[i];
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);
	VkCommandBufferBeginInfo begin_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);
	constexpr VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
	constexpr VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
	VkClearValue clear_values[] = {clear_color, clear_depth};
	VkRenderPassBeginInfo pass_info{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = veekay::app.vk_render_pass,
		.framebuffer = framebuffer,
		.renderArea = {.extent = {veekay::app.window_width, veekay::app.window_height}},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &pass_info, VK_SUBPASS_CONTENTS_INLINE);
	VkDeviceSize zero_offset = 0;
	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;
	
	if (!wireframe_mode) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			const Mesh& mesh = model.mesh;
			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}
			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}
			uint32_t offset = i * sizeof(ModelUniforms);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}
	} else {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe_pipeline);
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			const Mesh& mesh = model.mesh;
			
			// Set color for this cone's wireframe
			float edge_color[4] = {
				model.material.albedo.x,
				model.material.albedo.y,
				model.material.albedo.z,
				1.0f
			};
			vkCmdSetBlendConstants(cmd, edge_color);
			
			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}
			if (current_index_buffer != mesh.edge_buffer->buffer) {
				current_index_buffer = mesh.edge_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}
			uint32_t offset = i * sizeof(ModelUniforms);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
			vkCmdDrawIndexed(cmd, mesh.edge_indices, 1, 0, 0, 0);
		}
	}
	// float edge_color[4] = {0.0f, 0.0f, 1.0f, 1.0f};
	// vkCmdSetBlendConstants(cmd, edge_color);
	// vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe_pipeline);
	// for (size_t i = 0, n = models.size(); i < n; ++i) {
	// 	const Model& model = models[i];
	// 	const Mesh& mesh = model.mesh;
	// 	if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
	// 		current_vertex_buffer = mesh.vertex_buffer->buffer;
	// 		vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
	// 	}
	// 	if (current_index_buffer != mesh.edge_buffer->buffer) {
	// 		current_index_buffer = mesh.edge_buffer->buffer;
	// 		vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
	// 	}
	// 	uint32_t offset = i * sizeof(ModelUniforms);
	// 	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
	// 	vkCmdDrawIndexed(cmd, mesh.edge_indices, 1, 0, 0, 0);
	// }
	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

}

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,	
		.update = update,
		.render = render,
	});
}
