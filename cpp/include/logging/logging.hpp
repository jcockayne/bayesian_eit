//#include "logging/easylogging++.hpp"
#include <iostream>

#define LOG_MESSAGE(level, message) std::cout << level << " in " << __FUNCTION__ << " (" << __FILE__ << ":" << __LINE__ << ") : " << message << std::endl; 
#ifdef ENABLE_LOG_TRACE
	#define LOG_TRACE(x) do { \
		LOG_MESSAGE("TRACE", x) \
	} while (0)

#else
	#define LOG_TRACE(x) do {} while (0)
#endif

#ifdef ENABLE_LOG_DEBUG
	#define LOG_DEBUG(x) do { \
		LOG_MESSAGE("DEBUG", x) \
	} while (0)
#else
	#define LOG_DEBUG(x) do {} while (0)
#endif

#ifdef ENABLE_LOG_INFO
	#define LOG_INFO(x) do { \
		LOG_MESSAGE("INFO", x) \
	} while (0)
#else
	#define LOG_INFO(x) do {} while (0)
#endif

#ifdef ENABLE_LOG_WARN
	#define LOG_WARN(x) do { \
		LOG_MESSAGE("WARN", x) \
	} while (0)
#else
	#define LOG_WARN(x) do {} while (0)
#endif

#ifdef ENABLE_LOG_ERROR
	#define LOG_ERROR(x) do { \
		LOG_MESSAGE("ERROR", x) \
	} while (0)
#else
	#define LOG_ERROR(x) do {} while (0)
#endif