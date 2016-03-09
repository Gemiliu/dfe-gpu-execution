#include "timer.h"

Timer::Timer() : first(true)
{
}

void Timer::clear() {
	this->first = true;
	this->startIdentifiers.clear();
	this->beginTimes.clear();
	this->endTimes.clear();
	this->stackIdentifiers = std::stack<std::string>();
}

void Timer::begin(const std::string &msg) {
	if (this->first) {
		this->timer.start();
		this->first = false;
 	}
	this->stackIdentifiers.push(msg);
	this->beginTimes[msg] = this->timer.get_elapsed_time_mcs();
}

void Timer::end(const std::string &msg) {
	check(this->stackIdentifiers.size() != 0 && this->stackIdentifiers.top() == msg);
	startIdentifiers.push_back(msg);
	this->endTimes[msg] = this->timer.get_elapsed_time_mcs();
	this->stackIdentifiers.pop();
}

unsigned long long Timer::getTimeMicroseconds(const std::string &msg) {
	for (int i = 0; i < static_cast <int> (this->startIdentifiers.size()); i++) {
		if (this->startIdentifiers[i] == msg) {
			return this->endTimes[msg] - this->beginTimes[msg];
		}
	}
	throw std::string("no times are there: " + msg);
}

unsigned int Timer::getTimeMilliseconds(const std::string &msg) {
	return (unsigned int)(getTimeMicroseconds(msg) / 1e3);
}

float Timer::getTimeMillisecondsFloat(const std::string &msg) {
	return getTimeMicroseconds(msg) / static_cast<float>(1e3);
}

unsigned int Timer::getTimeSec(const std::string &msg) {
	return (unsigned int)(getTimeMicroseconds(msg) / 1e6);
}

float Timer::getTimeSecFloat(const std::string &msg) {
	return getTimeMicroseconds(msg) / static_cast<float>(1e6);
}
