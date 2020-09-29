#pragma once

#include <exception>
#include <sstream>
#include <fstream>
#include <string>

static void expect_token(std::string found, const char* expected)
{
	if (found != expected)
	{
		std::ostringstream ss;
		ss << "Expected '" << expected << "', found '" << found << "' instead.";
		auto msg = ss.str();
		throw std::exception(msg.c_str());
	}
}
static void unknown_token(std::string token)
{
	std::ostringstream ss;
	ss << "Unknown token '" << token << "'.";
	auto msg = ss.str();
	throw std::exception(msg.c_str());
}
static void bad_unit(int x, int y, const char* err)
{
	std::ostringstream ss;
	ss << "Bad unit at (" << x << ", " << y << ")";
	if (err != nullptr) ss << ": " << err;
	auto msg = ss.str();
	throw std::exception(msg.c_str());
}