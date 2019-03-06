#include "strlib.h"

int _vscprintf (const char * format, va_list pargs)
{ 
    int retval; 
    va_list argcopy;
    va_copy(argcopy, pargs); 
    retval = vsnprintf(NULL, 0, format, argcopy); 
    va_end(argcopy); 
    return retval;
}

string strsprintf(const char* format, ...)
{
	va_list args;
	va_start(args, format);

	int   len = _vscprintf(format, args) + 1;//_vscprintf doesn't count terminating '\0'
	char* buffer = new char[len];

	vsprintf(buffer, format, args);
	string retStr(buffer);


	delete[]buffer;
	va_end(args);

	return retStr;
}