#ifndef TIMER_H_
#define TIMER_H_

#include "check.h"
#include <stack>
#include <map>

#ifdef _WINDOWS_
#include <windows.h>
#else
#include <sys/time.h>

inline unsigned int get_tick_count() {
	struct timeval tv;
	if (gettimeofday(&tv, NULL) != 0) {
		return 0;
	}
	return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}

inline unsigned long long get_tick_count_mcs() {
	struct timeval tv;
	if (gettimeofday(&tv, NULL) != 0) {
		return 0;
	}
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#endif

class TimerMicrosecondsPrecision {
private:
#ifdef _WINDOWS_
	LARGE_INTEGER timer_value;
	LARGE_INTEGER frequency;
#else
	unsigned long long timer_value;
#endif

public:

	void start() {
		BEGIN_FUNCTION {
#ifdef _WINDOWS_
			check(QueryPerformanceCounter(&timer_value));
			check(QueryPerformanceFrequency(&frequency));

#else
			timer_value = get_tick_count_mcs();
#endif
		} END_FUNCTION
	}

	unsigned long long get_elapsed_time_mcs() {
		BEGIN_FUNCTION {
#ifdef _WINDOWS_
			LARGE_INTEGER curr_timer_value;
			check(QueryPerformanceCounter(&curr_timer_value));
			return static_cast <unsigned long long> ( static_cast <double> (curr_timer_value.QuadPart - timer_value.QuadPart) * 1000000.0 / static_cast <double> (frequency.QuadPart) );
#else
			unsigned long long curr_timer_value = get_tick_count_mcs();
			return static_cast <unsigned long long> (curr_timer_value - timer_value);
#endif
		} END_FUNCTION
	}
};

class Timer {
private:
	std::vector <std::string> startIdentifiers;
	std::map <std::string, unsigned long long> beginTimes;
	std::map <std::string, unsigned long long> endTimes;

	TimerMicrosecondsPrecision timer;
	bool first;
	std::stack<std::string> stackIdentifiers;
public:

	Timer();
	void clear();
	void begin(const std::string &msg);
	void end(const std::string &msg);

	unsigned int getTimeMilliseconds(const std::string &msg);
	float getTimeMillisecondsFloat(const std::string &msg);
	unsigned long long getTimeMicroseconds(const std::string &msg);
	unsigned int getTimeSec(const std::string &msg);
	float getTimeSecFloat(const std::string &msg);
};

#endif // TIMER_H_
