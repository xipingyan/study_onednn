#pragma once

#define LOG_POS std::cout << __FUNCTION__ << ":" << __LINE__
#define PERR LOG_POS << " Error: "
#define PWARNING LOG_POS << " Warning: "
#define PLOG LOG_POS << " Log: "