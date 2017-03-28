#include <stdlib.h>

typedef struct output
{
	int *text;
	int *amount;
} output;

void outputInit(output out, int size);
void outputDispose(output out);