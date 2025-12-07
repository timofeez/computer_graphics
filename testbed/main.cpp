#include "veekay/input.hpp" // Подключение заголовка для работы с вводом (клавиатура, мышь)
#include <format> // Подключение библиотеки форматирования строк (С++20)
#include <climits> // Подключение библиотеки для констант предельных значений типов
#include <vector> // Подключение контейнера динамического массива
#include <stack> // Подключение контейнера стека
#include <iostream> // Подключение потоков ввода-вывода (для логов)
#include <fstream> // Подключение файловых потоков (для чтения шейдеров)
#include <cmath> // Подключение математических функций (sin, cos, sqrt)
#include <veekay/veekay.hpp> // Подключение основного заголовка движка veekay
#include <imgui.h> // Подключение библиотеки графического интерфейса ImGui
#include <vulkan/vulkan_core.h> // Подключение ядра Vulkan API
#include <lodepng.h> // Подключение библиотеки для загрузки PNG изображений

namespace { // Анонимное пространство имен для локальных сущностей

constexpr uint32_t max_models = 1024; // Максимальное количество моделей в сцене
constexpr uint32_t max_point_lights = 32; // Максимальное количество точечных источников света

struct Vertex { // Структура вершины для передачи в шейдер
	veekay::vec3 position; // Позиция вершины (x, y, z)
	veekay::vec3 normal; // Нормаль к поверхности в этой вершине
	veekay::vec2 uv; // Текстурные координаты (u, v)
};

struct SceneUniforms { // Глобальные параметры сцены для шейдера
	veekay::mat4 view_projection; // Матрица Вид-Проекция камеры игрока
	veekay::vec3 camera_position; // Позиция камеры в мировых координатах
	float _padding1; // Выравнивание структуры до 16 байт (std140)
};

struct alignas(16) LightUniforms { // Параметры матрицы источника света
	veekay::mat4 light_view_projection; // Матрица Вид-Проекция для рендеринга карт теней
};

struct alignas(16) ModelUniforms { // Параметры конкретной модели
	veekay::mat4 model; // Матрица Модели (трансформация объекта)
	veekay::vec3 albedo_color; // Базовый цвет материала
	float use_texture; // Флаг использования текстуры (1.0 - да, 0.0 - нет)
	veekay::vec3 specular_color; // Цвет блика
	float shininess; // Сила блеска (степень в модели Фонга)
};

struct alignas(16) DirectionalLight { // Структура направленного источника света (Солнце)
	veekay::vec3 direction; // Направление света
	float _padding1; // Выравнивание
	veekay::vec3 color; // Цвет света
	float intensity; // Интенсивность
};

struct alignas(16) PointLight { // Структура точечного источника света
	veekay::vec3 position; // Позиция источника
	float _padding1; // Выравнивание
	veekay::vec3 color; // Цвет света
	float intensity; // Интенсивность
};

struct alignas(16) LightingUniforms { // Единый блок освещения для шейдера
	DirectionalLight directional; // Один направленный свет
	veekay::vec3 ambient_color; // Цвет фонового освещения
	float ambient_intensity; // Интенсивность фонового освещения
	uint32_t point_light_count; // Количество активных точечных источников
	float _padding2; // Выравнивание
	float _padding3;
	float _padding4;
};

struct Material { // Описание материала на стороне CPU
	veekay::vec3 albedo; // Основной цвет
	veekay::vec3 specular; // Цвет блика
	float shininess; // Блеск
};

struct Mesh { // Описание 3D-сетки (меша)
	veekay::graphics::Buffer* vertex_buffer; // Указатель на вершинный буфер GPU
	veekay::graphics::Buffer* edge_buffer; // Указатель на буфер ребер (для wireframe)
	veekay::graphics::Buffer* index_buffer; // Указатель на индексный буфер GPU
	uint32_t indices; // Количество индексов для рисования треугольников
	uint32_t edge_indices; // Количество индексов для рисования линий
};

struct Transform { // Компоненты трансформации объекта
	veekay::vec3 position = {}; // Позиция
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f}; // Масштаб
	veekay::vec3 rotation = {}; // Вращение (углы Эйлера в градусах)
	[[nodiscard]] veekay::mat4 matrix() const; // Метод получения итоговой матрицы 4x4
};

struct Model { // Сущность игрового объекта
	Mesh mesh; // Ссылка на геометрию
	Transform transform; // Данные о положении
	Material material; // Данные о материале
float rotation_speed = 1.0f; // Скорость автоматического вращения
	veekay::vec3 initial_rotation = {}; // Начальное вращение
};

struct Camera { // Описание камеры игрока
	constexpr static float default_fov = 60.0f; // Угол обзора по умолчанию
	constexpr static float default_near_plane = 0.01f; // Ближняя плоскость отсечения
	constexpr static float default_far_plane = 100.0f; // Дальняя плоскость отсечения
	veekay::vec3 position = {}; // Позиция камеры
	veekay::vec3 rotation = {}; // Вращение камеры
	float fov = default_fov; // Текущий угол обзора
	float near_plane = default_near_plane; // Текущая ближняя плоскость
	float far_plane = default_far_plane; // Текущая дальняя плоскость
	[[nodiscard]] veekay::mat4 view() const; // Получение матрицы вида
	[[nodiscard]] veekay::mat4 view_projection(float aspect_ratio) const; // Получение матрицы вид-проекция
};

inline namespace { // Глобальные переменные в анонимном пространстве имен
	Camera camera{ // Инициализация камеры
		.position = {8.0f, 6.0f, 8.0f}, // Начальная позиция
		.rotation = {35.0f, -180.0f, -180.0f} // Начальный поворот
	};
	std::vector<Model> models; // Вектор всех моделей в сцене
	float global_rotation_speed = 1.5f; // Множитель скорости вращения всех объектов
	bool wireframe_mode = false; // Флаг режима отображения "сетка"
	
	LightingUniforms lighting{ // Инициализация параметров освещения
		.directional = {
			.direction = veekay::vec3::normalized({-0.8f, -0.3f, -0.4f}), // Направление света (низкий угол для длинных теней)
			.color = {1.0f, 1.0f, 1.0f}, // Белый цвет
			.intensity = 8.0f // Высокая интенсивность
		},
		.ambient_color = {1.0f, 1.0f, 1.0f}, // Белый фоновый свет
		.ambient_intensity = 0.0f // Выключен, чтобы тени были контрастными
	};
	
	std::vector<PointLight> point_lights; // Вектор точечных источников
	uint32_t point_light_count = 0; // Счетчик точечных источников
	
	bool mouse_captured = false; // Флаг захвата мыши окном
	float camera_speed = 5.0f; // Скорость полета камеры
	float mouse_sensitivity = 0.1f; // Чувствительность мыши
}

inline namespace { // Переменные Vulkan ресурсов
	VkShaderModule vertex_shader_module; // Модуль вершинного шейдера
	VkShaderModule fragment_shader_module; // Модуль фрагментного шейдера
	VkDescriptorPool descriptor_pool; // Пул дескрипторов (ресурсов шейдера)
	VkDescriptorSetLayout descriptor_set_layout; // Схема набора дескрипторов
	VkDescriptorSet descriptor_set; // Набор дескрипторов (связывает буферы с шейдером)
		VkPipelineLayout pipeline_layout; // Схема конвейера
	VkPipeline pipeline; // Графический конвейер (основной)
	VkPipeline wireframe_pipeline; // Конвейер для отрисовки сетки
	veekay::graphics::Buffer* scene_uniforms_buffer; // Буфер глобальных униформ
	veekay::graphics::Buffer* model_uniforms_buffer; // Буфер униформ моделей
	veekay::graphics::Buffer* lighting_uniforms_buffer; // Буфер освещения
	veekay::graphics::Buffer* point_lights_buffer; // Буфер точечных источников
	veekay::graphics::Buffer* light_uniforms_buffer; // Буфер матрицы света (для теней)
	size_t model_uniform_stride = sizeof(ModelUniforms); // Шаг данных в буфере моделей
	Mesh cone_mesh; // Меш конуса
	Mesh plane_mesh; // Меш плоскости
	veekay::graphics::Texture* missing_texture; // Текстура-заглушка (розовая шашка)
	VkSampler missing_texture_sampler; // Сэмплер для заглушки
	veekay::graphics::Texture* lenna_texture; // Основная текстура
	VkSampler lenna_texture_sampler; // Сэмплер основной текстуры
	
	constexpr uint32_t shadow_map_size = 2048; // Разрешение карты теней
	VkImage shadow_map_image = VK_NULL_HANDLE; // Изображение карты теней
	VkDeviceMemory shadow_map_memory = VK_NULL_HANDLE; // Память карты теней
	VkImageView shadow_map_view = VK_NULL_HANDLE; // Представление изображения (View)
	VkSampler shadow_map_sampler = VK_NULL_HANDLE; // Сэмплер для чтения карты теней
	VkShaderModule shadow_vertex_shader_module = VK_NULL_HANDLE; // Вершинный шейдер прохода теней
	VkShaderModule shadow_fragment_shader_module = VK_NULL_HANDLE; // Фрагментный шейдер прохода теней
	VkPipeline shadow_pipeline = VK_NULL_HANDLE; // Конвейер рендеринга теней
	VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE; // Схема конвейера теней
	VkDescriptorSetLayout shadow_descriptor_set_layout = VK_NULL_HANDLE; // Схема дескрипторов теней
	VkDescriptorSet shadow_descriptor_set = VK_NULL_HANDLE; // Набор дескрипторов теней
	VkDescriptorPool shadow_descriptor_pool = VK_NULL_HANDLE; // Пул дескрипторов теней
}

float toRadians(float degrees) { // Вспомогательная функция перевода градусов в радианы
	return degrees * static_cast<float>(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const { // Вычисление матрицы модели (Scale * Rotation * Translation)
	const auto scaling_mtx = veekay::mat4::scaling(scale); // Матрица масштабирования
	const auto rot_mtx_x = veekay::mat4::rotation({1., .0, .0}, rotation.x); // Вращение по X
	const auto rot_mtx_y = veekay::mat4::rotation({.0, -1., .0}, rotation.y); // Вращение по Y
	const auto rot_mtx_z = veekay::mat4::rotation({.0, .0, 1.}, rotation.z); // Вращение по Z
	auto t = veekay::mat4::translation(position); // Матрица перемещения
	// Порядок перемножения важен для корректной композиции трансформаций
	return scaling_mtx * rot_mtx_x * rot_mtx_y * rot_mtx_z * t;
}

veekay::mat4 Camera::view() const { // Вычисление матрицы вида камеры (обратная к трансформации камеры)
	const auto t = veekay::mat4::translation(-position); // Обратное перемещение
	const auto rot_mtx_x = veekay::mat4::rotation({1., .0, .0}, toRadians(rotation.x));
	const auto rot_mtx_y = veekay::mat4::rotation({.0, -1., .0}, toRadians(rotation.y));
	const auto rot_mtx_z = veekay::mat4::rotation({.0, .0, 1.}, toRadians(rotation.z));
	return t * rot_mtx_x * rot_mtx_y * rot_mtx_z; // Сначала поворот, потом перемещение (для камеры)
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const { // Комбинированная матрица Вид * Проекция
	auto projection_mtx = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	return view() * projection_mtx;
}

veekay::mat4 calculateLightViewProjection(const veekay::vec3& light_direction, float size, float near_plane, float far_plane) { // Расчет матрицы света для теней
	// Создаем матрицу вида "от лица" источника света
	veekay::vec3 scene_center = {0.0f, 0.0f, 0.0f}; // Центр сцены, куда смотрит свет
	veekay::vec3 forward = veekay::vec3::normalized(light_direction); // Направление взгляда (совпадает с лучом света)
	veekay::vec3 light_pos = scene_center - forward * (size * 0.5f); // Виртуальная позиция источника
	
	veekay::vec3 up = {0.0f, 1.0f, 0.0f}; // Вектор "вверх"
	if (std::abs(veekay::vec3::dot(forward, up)) > 0.9f) { // Защита от коллинеарности (Gimbal lock)
		up = {1.0f, 0.0f, 0.0f};
	}
	veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(up, forward)); // Вектор "вправо"
	up = veekay::vec3::normalized(veekay::vec3::cross(forward, right)); // Пересчитанный ортогональный "вверх"
	
	// Ручное конструирование матрицы вида (LookAt)
	veekay::mat4 view = veekay::mat4::identity();
	view[0][0] = right.x;   view[1][0] = right.y;   view[2][0] = right.z;   view[3][0] = -veekay::vec3::dot(right, light_pos);
	view[0][1] = up.x;      view[1][1] = up.y;      view[2][1] = up.z;      view[3][1] = -veekay::vec3::dot(up, light_pos);
	view[0][2] = forward.x; view[1][2] = forward.y; view[2][2] = forward.z; view[3][2] = -veekay::vec3::dot(forward, light_pos);
	view[0][3] = 0.0f;      view[1][3] = 0.0f;      view[2][3] = 0.0f;      view[3][3] = 1.0f;
	
	// Ортогональная проекция для направленного света
	veekay::mat4 projection = veekay::mat4::orthographic(-size, size, -size, size, near_plane, far_plane);
	
	return view * projection; // Итоговая матрица света
}

std::vector<unsigned char> loadImage(const char* path, unsigned& width, unsigned& height) { // Загрузка изображения с диска
	std::vector<unsigned char> image;
	unsigned error = lodepng::decode(image, width, height, path); // Декодирование PNG
	if (error) {
		std::cerr << "Failed to load image " << path << ": " << lodepng_error_text(error) << "\n";
		return {};
	}
	return image; // Возвращает сырые пиксели
}

VkShaderModule loadShaderModule(const char* path) { // Загрузка SPIR-V шейдера
	std::ifstream file(path, std::ios::binary | std::ios::ate); // Открытие файла в бинарном режиме, курсор в конец
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << path << "\n";
		return nullptr;
	}
	std::streampos file_size = file.tellg(); // Получение размера
	if (file_size <= 0) {
		std::cerr << "Invalid shader file size: " << path << "\n";
		file.close();
		return nullptr;
	}
	size_t size = static_cast<size_t>(file_size);
	if (size % sizeof(uint32_t) != 0) { // Проверка на кратность 4 байтам (требование SPIR-V)
		std::cerr << "Shader file size is not a multiple of 4: " << path << "\n";
		file.close();
		return nullptr;
	}
	std::vector<uint32_t> buffer(size / sizeof(uint32_t)); // Буфер для кода
	file.seekg(0); // Курсор в начало
	file.read(reinterpret_cast<char*>(buffer.data()), size); // Чтение
	file.close();
	if (file.gcount() != static_cast<std::streamsize>(size)) {
		std::cerr << "Failed to read entire shader file: " << path << "\n";
		return nullptr;
	}
	VkShaderModuleCreateInfo info{ // Структура создания модуля
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};
	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) { // Создание модуля
		return nullptr;
	}
	return result;
}

void initialize(VkCommandBuffer cmd) { // Функция инициализации (вызывается один раз при старте)
	VkDevice& device = veekay::app.vk_device; // Ссылка на логическое устройство
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device; // Ссылка на физическое устройство
	{
		// Загрузка шейдеров
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
		// Описание стадий шейдеров для конвейера
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
		// Описание входных данных вершин (Binding)
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};
		// Описание атрибутов вершин (Позиция, Нормаль, UV)
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
		// Сборка описания входных данных
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = std::size(attributes),
			.pVertexAttributeDescriptions = attributes,
		};
		// Сборка примитивов (треугольники)
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};
		// Настройка растеризации (заливка полигонов)
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT, // Отсечение задних граней
			.frontFace = VK_FRONT_FACE_CLOCKWISE, // Порядок обхода по часовой
			.lineWidth = 1.0f,
		};
		// Настройка мультисэмплинга (выключен, 1 сэмпл)
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};
		// Настройка вьюпорта (области рисования)
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
		// Настройка теста глубины (Depth Test)
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true, // Запись в буфер глубины разрешена
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};
		// Настройка смешивания цветов (Blending) - выключено для основного прохода
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
		// Настройка смешивания для wireframe (включено аддитивное смешивание)
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
			// Описание размеров пула дескрипторов (сколько ресурсов каждого типа мы можем выделить)
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
			},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
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
			// Описание привязок (Bindings) в наборе дескрипторов
			VkDescriptorSetLayoutBinding bindings[] = {
				{ // Сцена
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{ // Модель (динамический униформ)
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{ // Освещение
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{ // Точечные источники (Storage Buffer)
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{ // Текстура модели
					.binding = 4,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{ // Карта теней
					.binding = 5,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{ // Матрица света
					.binding = 6,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
			// Выделение набора дескрипторов из пула
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
		// Создание Pipeline Layout (интерфейс конвейера)
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
		// Создание основного графического конвейера
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
		// Создание конвейера для Wireframe (сетки)
		VkPipelineInputAssemblyStateCreateInfo line_assembly_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		};
		VkPipelineRasterizationStateCreateInfo line_raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE, // Без отсечения граней
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
	// Создание буферов униформ (выделение памяти на CPU/GPU)
	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	model_uniform_stride = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * model_uniform_stride,
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
	light_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(LightUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	{
		// Создание сэмплера для отсутствующей текстуры
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};
		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}
		// Создание текстуры-заглушки (2x2 пикселя, розово-черная)
		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};
		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	{
		// Загрузка текстуры "Lenna" (или другой)
		unsigned width, height;
		std::vector<unsigned char> image_data = loadImage("assets/old_texture.png", width, height);
		if (image_data.empty()) {
			std::cerr << "Failed to load old_texture.png, using missing texture\n";
			lenna_texture = missing_texture;
			lenna_texture_sampler = missing_texture_sampler;
		} else {
			// Конвертация RGBA в BGRA (для Vulkan)
			std::vector<uint32_t> bgra_pixels(width * height);
			for (size_t i = 0; i < width * height; ++i) {
				unsigned char r = image_data[i * 4 + 0];
				unsigned char g = image_data[i * 4 + 1];
				unsigned char b = image_data[i * 4 + 2];
				unsigned char a = image_data[i * 4 + 3];
				bgra_pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
			}
			lenna_texture = new veekay::graphics::Texture(cmd, width, height,
			                                             VK_FORMAT_B8G8R8A8_UNORM,
			                                             bgra_pixels.data());
			
			// Создание сэмплера для текстуры (линейная фильтрация, повторение координат)
			VkSamplerCreateInfo sampler_info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = VK_FILTER_LINEAR,
				.minFilter = VK_FILTER_LINEAR,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.mipLodBias = 0.0f,
				.anisotropyEnable = VK_TRUE,
				.maxAnisotropy = 16.0f,
				.compareEnable = VK_FALSE,
				.compareOp = VK_COMPARE_OP_ALWAYS,
				.minLod = 0.0f,
				.maxLod = 16.0f,
				.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
				.unnormalizedCoordinates = VK_FALSE,
			};
			if (vkCreateSampler(device, &sampler_info, nullptr, &lenna_texture_sampler) != VK_SUCCESS) {
				std::cerr << "Failed to create old_texture sampler\n";
				veekay::app.running = false;
				return;
			}
		}
	}
	{
		// Обновление дескрипторов (привязка буферов и текстур к слотам шейдера)
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
		VkDescriptorImageInfo image_info{
			.sampler = lenna_texture_sampler,
			.imageView = lenna_texture->view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
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
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_info,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 6,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
		};

		VkDescriptorBufferInfo light_buffer_info{
			.buffer = light_uniforms_buffer->buffer,
			.offset = 0,
			.range = sizeof(LightUniforms),
		};
		write_infos[5].pBufferInfo = &light_buffer_info;
		vkUpdateDescriptorSets(device, std::size(write_infos), write_infos, 0, nullptr);
	}
	{
		// Генерация меша Конуса
		constexpr float radius = 1.0f;
		constexpr float height = 2.0f;
		constexpr int segments = 32;
		std::vector<Vertex> vertices;
		// Верхняя вершина
		vertices.push_back(Vertex{
			.position = {0.0f, height, 0.0f},
			.normal = {0.0f, 1.0f, 0.0f},
			.uv = {0.5f, 0.0f}
		});
		// Боковые вершины
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
		// Центральная вершина основания
		vertices.push_back(Vertex{
			.position = {0.0f, 0.0f, 0.0f},
			.normal = {0.0f, -1.0f, 0.0f},
			.uv = {0.5f, 0.5f}
		});
		std::vector<uint32_t> indices;
		// Индексы боковой поверхности
		for (int i = 0; i < segments; ++i) {
			indices.push_back(0);
			indices.push_back(i + 1);
			indices.push_back((i + 1) % segments + 1);
		}
		// Индексы основания
		for (int i = 0; i < segments; ++i) {
			indices.push_back(base_center);
			indices.push_back(i + 1);
			indices.push_back((i + 1) % segments + 1);
		}
		std::vector<uint32_t> edge_indices;
		// Индексы для wireframe
		for (int i = 1; i <= segments; ++i) {
			edge_indices.push_back(0);
			edge_indices.push_back(i);
		}
		for (int i = 1; i <= segments; ++i) {
			edge_indices.push_back(i);
			edge_indices.push_back((i % segments) + 1);
		}
		// Создание GPU буферов для меша
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
	
	{
		// Генерация меша Плоскости
		constexpr float plane_size = 20.0f;
		std::vector<Vertex> vertices = {
			// Bottom-left
			Vertex{
				.position = {-plane_size, 0.0f, -plane_size},
				.normal = {0.0f, 1.0f, 0.0f},
				.uv = {0.0f, 0.0f}
			},
			// Bottom-right
			Vertex{
				.position = {plane_size, 0.0f, -plane_size},
				.normal = {0.0f, 1.0f, 0.0f},
				.uv = {1.0f, 0.0f}
			},
			// Top-right
			Vertex{
				.position = {plane_size, 0.0f, plane_size},
				.normal = {0.0f, 1.0f, 0.0f},
				.uv = {1.0f, 1.0f}
			},
			// Top-left
			Vertex{
				.position = {-plane_size, 0.0f, plane_size},
				.normal = {0.0f, 1.0f, 0.0f},
				.uv = {0.0f, 1.0f}
			}
		};
		std::vector<uint32_t> indices = {
			0, 1, 2,
			0, 2, 3
		};
		std::vector<uint32_t> edge_indices = {
			0, 1, 1, 2, 2, 3, 3, 0
		};
		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		plane_mesh.edge_buffer = new veekay::graphics::Buffer(
			edge_indices.size() * sizeof(uint32_t), edge_indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		plane_mesh.indices = static_cast<uint32_t>(indices.size());
		plane_mesh.edge_indices = static_cast<uint32_t>(edge_indices.size());
	}
	
	{
		// Настройка Shadow Map (Карты теней)
		VkFormat depth_format = VK_FORMAT_UNDEFINED;
		VkFormat candidates[] = {
			VK_FORMAT_D32_SFLOAT, // 32-бит float глубина
			VK_FORMAT_D32_SFLOAT_S8_UINT,
			VK_FORMAT_D24_UNORM_S8_UINT,
		};
		// Поиск подходящего формата глубины, который поддерживается как attachment и для сэмплинга
		for (const auto& f : candidates) {
			VkFormatProperties properties;
			vkGetPhysicalDeviceFormatProperties(physical_device, f, &properties);
			if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
			    (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
				depth_format = f;
				std::cerr << "Selected shadow map format: " << f << "\n";
				break;
			}
		}
		if (depth_format == VK_FORMAT_UNDEFINED) {
			std::cerr << "Failed to find suitable depth format for shadow map\n";
			veekay::app.running = false;
			return;
		}
		
		// Создание изображения карты теней
		VkImageCreateInfo image_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = depth_format,
			.extent = {shadow_map_size, shadow_map_size, 1},
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		if (vkCreateImage(device, &image_info, nullptr, &shadow_map_image) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow map image\n";
			veekay::app.running = false;
			return;
		}
		
		// Выделение памяти для текстуры
		VkMemoryRequirements mem_requirements;
		vkGetImageMemoryRequirements(device, shadow_map_image, &mem_requirements);
		
		VkPhysicalDeviceMemoryProperties mem_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
		
		uint32_t mem_type_index = UINT_MAX;
		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
			if ((mem_requirements.memoryTypeBits & (1 << i)) &&
			    (mem_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				mem_type_index = i;
				break;
			}
		}
		
		if (mem_type_index == UINT_MAX) {
			std::cerr << "Failed to find memory type for shadow map\n";
			veekay::app.running = false;
			return;
		}
		
		VkMemoryAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = mem_requirements.size,
			.memoryTypeIndex = mem_type_index,
		};
		if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow_map_memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate shadow map memory\n";
			veekay::app.running = false;
			return;
		}
		
		if (vkBindImageMemory(device, shadow_map_image, shadow_map_memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind shadow map memory\n";
			veekay::app.running = false;
			return;
		}
		
		// Создание View для карты теней
		VkImageViewCreateInfo view_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = shadow_map_image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = depth_format,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		if (vkCreateImageView(device, &view_info, nullptr, &shadow_map_view) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow map view\n";
			veekay::app.running = false;
			return;
		}
		
		// Создание сэмплера для теней (без сравнения, сравнение делаем в шейдере для PCF)
		VkSamplerCreateInfo sampler_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 1.0f,
			.compareEnable = VK_FALSE, // Ручное сравнение глубины в шейдере
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = 1.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE, // Вне карты теней = 1.0 (свет)
			.unnormalizedCoordinates = VK_FALSE,
		};
		if (vkCreateSampler(device, &sampler_info, nullptr, &shadow_map_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow map sampler\n";
			veekay::app.running = false;
			return;
		}
	}
	
	{
		// Инициализация конвейера рендеринга теней
		shadow_vertex_shader_module = loadShaderModule("shaders/shadow.vert.spv");
		if (!shadow_vertex_shader_module) {
			std::cerr << "Failed to load shadow vertex shader\n";
			veekay::app.running = false;
			return;
		}
		shadow_fragment_shader_module = loadShaderModule("shaders/shadow.frag.spv");
		if (!shadow_fragment_shader_module) {
			std::cerr << "Failed to load shadow fragment shader\n";
			veekay::app.running = false;
			return;
		}
		
		VkPipelineShaderStageCreateInfo shadow_stages[2] = {
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = shadow_vertex_shader_module,
				.pName = "main",
			},
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = shadow_fragment_shader_module,
				.pName = "main",
			},
		};
		
		VkVertexInputBindingDescription shadow_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};
		VkVertexInputAttributeDescription shadow_attributes[] = {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			},
		};
		VkPipelineVertexInputStateCreateInfo shadow_input_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &shadow_binding,
			.vertexAttributeDescriptionCount = 1,
			.pVertexAttributeDescriptions = shadow_attributes,
		};
		
		VkPipelineInputAssemblyStateCreateInfo shadow_assembly{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};
		
		VkViewport shadow_viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(shadow_map_size),
			.height = static_cast<float>(shadow_map_size),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		VkRect2D shadow_scissor{
			.offset = {0, 0},
			.extent = {shadow_map_size, shadow_map_size},
		};
		VkPipelineViewportStateCreateInfo shadow_viewport_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &shadow_viewport,
			.scissorCount = 1,
			.pScissors = &shadow_scissor,
		};
		
		// Растеризация теней
		VkPipelineRasterizationStateCreateInfo shadow_raster{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_FRONT_BIT, // Cull Front Faces -> Render Back Faces. Решает проблему "Acne".
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_TRUE,
			.depthBiasConstantFactor = 1.25f, // Смещение геометрии для борьбы с Peter Panning
			.depthBiasClamp = 0.0f,
			.depthBiasSlopeFactor = 1.75f,
			.lineWidth = 1.0f,
		};
		
		VkPipelineMultisampleStateCreateInfo shadow_multisample{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		};
		
		VkPipelineDepthStencilStateCreateInfo shadow_depth{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS, // Храним ближайшую к свету глубину
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
		};
		
		VkDescriptorSetLayoutBinding shadow_bindings[] = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			},
		};
		VkDescriptorSetLayoutCreateInfo shadow_layout_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = std::size(shadow_bindings),
			.pBindings = shadow_bindings,
		};
		if (vkCreateDescriptorSetLayout(device, &shadow_layout_info, nullptr, &shadow_descriptor_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow descriptor set layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkDescriptorPoolSize shadow_pool_sizes[] = {
			{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
			},
			{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.descriptorCount = 1,
			},
		};
		VkDescriptorPoolCreateInfo shadow_pool_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = 1,
			.poolSizeCount = std::size(shadow_pool_sizes),
			.pPoolSizes = shadow_pool_sizes,
		};
		if (vkCreateDescriptorPool(device, &shadow_pool_info, nullptr, &shadow_descriptor_pool) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow descriptor pool\n";
			veekay::app.running = false;
			return;
		}
		
		VkDescriptorSetAllocateInfo shadow_alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = shadow_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &shadow_descriptor_set_layout,
		};
		if (vkAllocateDescriptorSets(device, &shadow_alloc_info, &shadow_descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate shadow descriptor set\n";
			veekay::app.running = false;
			return;
		}
		
		VkDescriptorBufferInfo shadow_light_buffer{
			.buffer = light_uniforms_buffer->buffer,
			.offset = 0,
			.range = sizeof(LightUniforms),
		};
		VkDescriptorBufferInfo shadow_model_buffer{
			.buffer = model_uniforms_buffer->buffer,
			.offset = 0,
			.range = sizeof(ModelUniforms),
		};
		VkWriteDescriptorSet shadow_writes[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = shadow_descriptor_set,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &shadow_light_buffer,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = shadow_descriptor_set,
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &shadow_model_buffer,
			},
		};
		vkUpdateDescriptorSets(device, std::size(shadow_writes), shadow_writes, 0, nullptr);
		
		VkPipelineLayoutCreateInfo shadow_layout_create{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &shadow_descriptor_set_layout,
		};
		if (vkCreatePipelineLayout(device, &shadow_layout_create, nullptr, &shadow_pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkFormat shadow_depth_format = VK_FORMAT_UNDEFINED;
		VkFormat shadow_candidates[] = {
			VK_FORMAT_D32_SFLOAT,
			VK_FORMAT_D32_SFLOAT_S8_UINT,
			VK_FORMAT_D24_UNORM_S8_UINT,
		};
		for (const auto& f : shadow_candidates) {
			VkFormatProperties properties;
			vkGetPhysicalDeviceFormatProperties(physical_device, f, &properties);
			if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
				shadow_depth_format = f;
				break;
			}
		}
		if (shadow_depth_format == VK_FORMAT_UNDEFINED) {
			std::cerr << "Failed to find suitable depth format for shadow pipeline\n";
			veekay::app.running = false;
			return;
		}
		
		// Dynamic Rendering Info для теней
		VkPipelineRenderingCreateInfo shadow_rendering_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.depthAttachmentFormat = shadow_depth_format,
		};
		
		VkGraphicsPipelineCreateInfo shadow_pipeline_info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.pNext = &shadow_rendering_info,
			.stageCount = 2,
			.pStages = shadow_stages,
			.pVertexInputState = &shadow_input_state,
			.pInputAssemblyState = &shadow_assembly,
			.pViewportState = &shadow_viewport_state,
			.pRasterizationState = &shadow_raster,
			.pMultisampleState = &shadow_multisample,
			.pDepthStencilState = &shadow_depth,
			.layout = shadow_pipeline_layout,
		};
		if (vkCreateGraphicsPipelines(device, nullptr, 1, &shadow_pipeline_info, nullptr, &shadow_pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow pipeline\n";
			veekay::app.running = false;
			return;
		}
	}
	
	{
		// Привязка карты теней к основному шейдеру
		VkDescriptorImageInfo shadow_image_info{
			.sampler = shadow_map_sampler,
			.imageView = shadow_map_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
		};
		VkWriteDescriptorSet shadow_write{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 5,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &shadow_image_info,
		};
		vkUpdateDescriptorSets(device, 1, &shadow_write, 0, nullptr);
	}
	
	veekay::input::mouse::setCaptured(mouse_captured);
	
	// Создание точечных источников света (для красоты)
	point_lights.emplace_back(PointLight{
		.position = {2.0f, 5.0f, 2.0f},
		.color = {1.0f, 1.0f, 1.0f},
		.intensity = 0.0f // Выключен
	});
	point_lights.emplace_back(PointLight{
		.position = {-2.0f, 5.0f, -2.0f},
		.color = {1.0f, 0.5f, 0.3f},
		.intensity = 0.0f // Выключен
	});
	point_light_count = static_cast<uint32_t>(point_lights.size());

	// Создание моделей сцены
	// Плоскость
	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{
			.position = {0.0f, 0.0f, 0.0f},
			.scale = {1.0f, 1.0f, 1.0f},
			.rotation = {0.0f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {0.5f, 0.5f, 0.5f}, // Серый цвет
			.specular = {0.0f, 0.0f, 0.0f}, // Без блика
			.shininess = 1.0f
		},
		.rotation_speed = 0.0f,
		.initial_rotation = {0.0f, 0.0f, 0.0f}
	});
	
	// Конусы с разными цветами и размерами
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {-3.5f, 0.3f, 2.0f},
			.scale = {0.8f, 1.2f, 0.8f},
			.rotation = {0.0f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {0.2f, 0.8f, 0.2f}, // Зеленый
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 64.0f
		},
		.rotation_speed = 1.0f,
		.initial_rotation = {0.0f, 0.0f, 0.0f}
	});
	
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {-1.0f, 0.3f, -1.0f},
			.scale = {1.0f, 1.8f, 1.0f},
			.rotation = {0.0f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {0.2f, 0.2f, 0.9f}, // Синий
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 128.0f
		},
		.rotation_speed = 1.2f,
		.initial_rotation = {0.0f, 0.0f, 0.0f}
	});
	
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {1.5f, 0.3f, 1.0f},
			.scale = {1.2f, 2.5f, 1.2f},
			.rotation = {0.0f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {0.9f, 0.3f, 0.3f}, // Красный
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 64.0f
		},
		.rotation_speed = 0.8f,
		.initial_rotation = {0.0f, 0.0f, 0.0f}
	});
	
	models.emplace_back(Model{
		.mesh = cone_mesh,
		.transform = Transform{
			.position = {4.5f, 0.3f, -2.0f},
			.scale = {1.4f, 3.0f, 1.4f},
			.rotation = {0.0f, 0.0f, 0.0f}
		},
		.material = {
			.albedo = {0.9f, 0.9f, 0.2f}, // Желтый
			.specular = {0.8f, 0.8f, 0.8f},
			.shininess = 32.0f
		},
		.rotation_speed = 0.6f,
		.initial_rotation = {0.0f, 0.0f, 0.0f}
	});
}

void shutdown() { // Очистка ресурсов при выходе
	VkDevice& device = veekay::app.vk_device;
	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;
	if (lenna_texture != missing_texture) {
		vkDestroySampler(device, lenna_texture_sampler, nullptr);
		delete lenna_texture;
	}
	// Очистка ресурсов теней
	vkDestroySampler(device, shadow_map_sampler, nullptr);
	vkDestroyImageView(device, shadow_map_view, nullptr);
	vkDestroyImage(device, shadow_map_image, nullptr);
	vkFreeMemory(device, shadow_map_memory, nullptr);
	// Очистка мешей и буферов
	delete cone_mesh.index_buffer;
	delete cone_mesh.vertex_buffer;
	delete cone_mesh.edge_buffer;
	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;
	delete plane_mesh.edge_buffer;
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;
	delete lighting_uniforms_buffer;
	delete point_lights_buffer;
	delete light_uniforms_buffer;
	// Очистка Vulkan объектов
	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(device, shadow_descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, shadow_descriptor_pool, nullptr);
	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipeline(device, wireframe_pipeline, nullptr);
	vkDestroyPipeline(device, shadow_pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	vkDestroyShaderModule(device, shadow_fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);
}

veekay::vec3 getForwardVector(const veekay::vec3& rotation) { // Вектор вперед для камеры
	float pitch = toRadians(rotation.x);
	float yaw = toRadians(rotation.y);
	return veekay::vec3{
		// Стандартная FPS камера: движение только по XZ
		sinf(yaw),
		0.0f,
		cosf(yaw)
	};
}

veekay::vec3 getRightVector(const veekay::vec3& rotation) { // Вектор вправо для камеры
	float yaw = toRadians(rotation.y);
	return veekay::vec3{
		sinf(yaw - M_PI / 2.0f),
		0.0f,
		cosf(yaw - M_PI / 2.0f)
	};
}

veekay::vec3 getUpVector(const veekay::vec3& forward, const veekay::vec3& right) { // Вектор вверх (ортогональный)
	return veekay::vec3::normalized(veekay::vec3::cross(forward, right));
}

void update(double time) { // Логика обновления (вызывается каждый кадр)
	static double last_time = time;
	double delta_time = time - last_time; // Время прошедшее с прошлого кадра
	last_time = time;
	
	// Переключение захвата мыши по ESC
	if (veekay::input::keyboard::isKeyPressed(veekay::input::keyboard::Key::escape)) {
		mouse_captured = !mouse_captured;
		veekay::input::mouse::setCaptured(mouse_captured);
		veekay::input::mouse::resetCursorDelta();
	}
	
	// Управление камерой
	if (mouse_captured) {
		veekay::vec2 mouse_delta = veekay::input::mouse::cursorDelta();
		camera.rotation.y -= mouse_delta.x * mouse_sensitivity;
		camera.rotation.x -= mouse_delta.y * mouse_sensitivity;
		
		if (camera.rotation.x > 89.0f) camera.rotation.x = 89.0f; // Ограничение взгляда вверх/вниз
		if (camera.rotation.x < -89.0f) camera.rotation.x = -89.0f;
		
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
	
	// Отрисовка GUI
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
	// Нормализация убрана, чтобы UI не конфликтовал с вводом
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
	
	// Вращение моделей
	for (auto& model : models) {
		if (model.rotation_speed == 0.0f) {
			model.transform.rotation = model.initial_rotation;
			continue;
		}
		model.transform.rotation.x = model.initial_rotation.x;
		model.transform.rotation.y = model.initial_rotation.y + model.rotation_speed * global_rotation_speed * time;
		model.transform.rotation.z = model.initial_rotation.z;
	}
	
	float aspect_ratio = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height);
	
	// Обновление униформ
	SceneUniforms* scene_uniforms = static_cast<SceneUniforms*>(scene_uniforms_buffer->mapped_region);
	scene_uniforms->view_projection = camera.view_projection(aspect_ratio);
	scene_uniforms->camera_position = camera.position;
	
	char* model_uniforms_base = static_cast<char*>(model_uniforms_buffer->mapped_region);
	for (size_t i = 0; i < models.size(); ++i) {
		auto* model_uniforms = reinterpret_cast<ModelUniforms*>(model_uniforms_base + i * model_uniform_stride);
		model_uniforms->model = models[i].transform.matrix();
		model_uniforms->albedo_color = models[i].material.albedo;
		model_uniforms->use_texture = (i == 0) ? 0.0f : 1.0f; // Плоскость без текстуры
		model_uniforms->specular_color = models[i].material.specular;
		model_uniforms->shininess = models[i].material.shininess;
	}
	
	LightingUniforms* lighting_data = static_cast<LightingUniforms*>(lighting_uniforms_buffer->mapped_region);
	*lighting_data = lighting;
	lighting_data->point_light_count = point_light_count;
	
	PointLight* lights_data = static_cast<PointLight*>(point_lights_buffer->mapped_region);
	for (size_t i = 0; i < point_lights.size() && i < max_point_lights; ++i) {
		lights_data[i] = point_lights[i];
	}
	
	// Расчет матрицы света для теней
	LightUniforms* light_uniforms = static_cast<LightUniforms*>(light_uniforms_buffer->mapped_region);
	veekay::vec3 light_dir = veekay::vec3::normalized(lighting.directional.direction);
	light_uniforms->light_view_projection = calculateLightViewProjection(
		light_dir,
		50.0f, // Размер области тени
		0.1f,  // Near
		200.0f // Far
	);
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) { // Основной цикл рендеринга
	if (cmd == VK_NULL_HANDLE || framebuffer == VK_NULL_HANDLE) {
		return;
	}
	
	vkResetCommandBuffer(cmd, 0); // Сброс командного буфера
	VkCommandBufferBeginInfo begin_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
		return;
	}
	
	// 1. ПРОХОД ТЕНЕЙ (Shadow Pass)
	if (shadow_map_image != VK_NULL_HANDLE && shadow_pipeline != VK_NULL_HANDLE && 
	    shadow_map_view != VK_NULL_HANDLE && shadow_descriptor_set != VK_NULL_HANDLE &&
	    shadow_pipeline_layout != VK_NULL_HANDLE) {
		
		static bool first_shadow_render = true;
		// Барьер памяти для перехода изображения в статус доступного для записи глубины
		VkImageMemoryBarrier shadow_barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = first_shadow_render ? static_cast<VkAccessFlags>(0) : VK_ACCESS_SHADER_READ_BIT,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = first_shadow_render ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_map_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			0, 0, nullptr, 0, nullptr, 1, &shadow_barrier);
		
		// Настройка аттачмента глубины (очистка перед рендерингом)
		VkRenderingAttachmentInfo depth_attachment{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = shadow_map_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {.depthStencil = {1.0f, 0}}, // Очистка в 1.0 (максимальная глубина)
		};
		
		// Начало динамического рендеринга
		VkRenderingInfo rendering_info{
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.renderArea = {.extent = {shadow_map_size, shadow_map_size}},
			.layerCount = 1,
			.colorAttachmentCount = 0,
			.pDepthAttachment = &depth_attachment,
		};
		
		vkCmdBeginRendering(cmd, &rendering_info);
		
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
		
		VkDeviceSize zero_offset = 0;
		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;
		
		if (!light_uniforms_buffer) {
			std::cerr << "ERROR: light_uniforms_buffer is null!\n";
			vkCmdEndRendering(cmd);
			return;
		}
		
		// Отрисовка моделей в карту теней
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			// Пропускаем плоскость в проходе теней, чтобы избежать артефактов на полу
			if (i == 0) continue;

			const Model& model = models[i];
			const Mesh& mesh = model.mesh;
			
			if (!mesh.vertex_buffer || !mesh.index_buffer || mesh.indices == 0) {
				continue;
			}
			
			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}
			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}
			
			uint32_t offsets[] = {static_cast<uint32_t>(i * model_uniform_stride)};
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout, 0, 1, &shadow_descriptor_set, 1, offsets);
			
			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}
		
		vkCmdEndRendering(cmd);
		
		first_shadow_render = false;
		
		// Барьер памяти для перевода карты теней в режим чтения шейдером
		VkImageMemoryBarrier shadow_barrier_back{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_map_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &shadow_barrier_back);
		
		VkMemoryBarrier mem_barrier{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
	}
	
	// 2. ОСНОВНОЙ ПРОХОД (Main Pass)
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
			uint32_t offset = static_cast<uint32_t>(i * model_uniform_stride);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}
	} else {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe_pipeline);
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			const Mesh& mesh = model.mesh;
			
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
			uint32_t offset = static_cast<uint32_t>(i * model_uniform_stride);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
			vkCmdDrawIndexed(cmd, mesh.edge_indices, 1, 0, 0, 0);
		}
	}

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
