 /// config.h
#pragma once

#ifdef _DEBUG
	#define MLCORE_DEBUG
#endif /// _DEBUG

// Platform detection
#ifdef _WIN32
	#define MLCORE_WINDOWS
#elifdef __linux__
	#define MLCORE_LINUX
#elifdef __APPLE__
	#define MLCORE_MACOS
#endif

/// SIMD Alignment (32 bits)
#define MLCORE_SIMD_ALIGNMENT 32