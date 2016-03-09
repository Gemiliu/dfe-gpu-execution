#ifndef CHECK_H_
#define CHECK_H_

#include <string>
#include <vector>
#include <limits>
#include <time.h>

#ifdef _MSC_VER
#include <windows.h>
#undef min
#undef max
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif

inline void throwString(std::string fileName, std::string lineNumber, std::string failedExpression) {
#ifdef _MSC_VER
	if (IsDebuggerPresent()) {
		DebugBreak();
	}
#endif
	throw "Check failed: " + failedExpression + ", file " + fileName + ", line " + lineNumber;
}

inline void throwString(std::string fileName, std::string lineNumber, std::string failedExpression, std::string comment) {
#ifdef _MSC_VER
	if (IsDebuggerPresent()) {
		DebugBreak();
	}
#endif
	throw "Check failed: " + failedExpression + " (" + comment + "), file " + fileName + ", line " + lineNumber;
}

#ifdef QUOTE
#error QUOTE macro defined not only in check.h
#endif

#ifdef QUOTE_VALUE
#error QUOTE_VALUE macro defined not only in check.h
#endif

#ifdef check
#error check macro defined not only in check.h
#endif

#define QUOTE(x) #x
#define QUOTE_VALUE(x) QUOTE(x)
#define check(expression, ...) \
{ \
	if (!(expression)) { \
		throwString(__FILE__, QUOTE_VALUE(__LINE__), #expression, ##__VA_ARGS__); \
	} \
}

#ifdef _MSC_VER
#define CURRENT_FUNCTION __FUNCSIG__
#elif __GNUG__
#define CURRENT_FUNCTION __PRETTY_FUNCTION__
#else
#define CURRENT_FUNCTION __FILE__ "(" QUOTE_VALUE(__LINE__) ")"
#endif

#define BEGIN_FUNCTION \
	try

#define END_FUNCTION \
	catch (const std::string &message) { \
		throw CURRENT_FUNCTION + (" " + message); \
	}

#define BEGIN_DESTRUCTOR \
	try { \
		try

#define END_DESTRUCTOR \
		catch (std::string message) { \
			ExceptionReporterList::reportException(__FILE__, QUOTE_VALUE(__LINE__), message); \
		} catch (...) { \
			ExceptionReporterList::reportException(__FILE__, QUOTE_VALUE(__LINE__), "unknown exception"); \
		} \
	} catch (...) {}

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif // CHECK_H_
