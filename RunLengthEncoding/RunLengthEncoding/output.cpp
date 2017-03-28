#include "output.h"

void outputInit(output out, int size)
{
	if(out.text == NULL)
	{
		out.text = (int*)malloc(size * sizeof(int));
	}
	if(out.amount == NULL)
	{
		out.amount = (int*)malloc(size * sizeof(int));
	}
}

void outputDispose(output out)
{
	if(out.text != NULL)
	{
		free(out.text);
	}
	if(out.amount != NULL)
	{
		free(out.amount);
	}
}