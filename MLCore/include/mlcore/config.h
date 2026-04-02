// config.h
#pragma once

#ifdef _DEBUG
	#define ML_CORE_DEBUG
#endif // _DEBUG

// Platform detection
#ifdef _WIN32
	#define ML_CORE_WINDOWS
#elifdef __linux__
	#define ML_CORE_LINUX
#elifdef __APPLE__
	#define ML_CORE_MACOS
#endif

// SIMD Alignment (32 bits)
#define ML_CORE_SIMD_ALIGNMENT 32