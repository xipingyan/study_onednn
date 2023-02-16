#pragma once

#define LOG_POS __FUNCTION__ << ":" << __LINE__
#define PERR std::cout << "Error: " << LOG_POS << " "
#define PWARNING std::cout << "Warning: " << LOG_POS << " "
#define PLOG std::cout << "Log: " << LOG_POS << " "