#include <cstdio>
#include <ctime>
#include <cstdlib>

#define INPUT_SIZE 64000000

int initializeArray(int **arr, int size, bool fillRandom)
{
	*arr = (int*)malloc(size * sizeof(int));

	if(*arr == NULL) {
		return 1;
	}

	if(fillRandom) {
		srand(time(NULL));
		for(int i = 0; i < size; i++) {
			(*arr)[i] = rand() % 5 + 63;
		}
	} else {
		for(int i = 0; i < size; i++) {
			(*arr)[i] = 0;
		}
		return 0;
	}
	return 0;
}


int initializeCharArray(char **arr, int size, bool fillRandom)
{
	*arr = (char*)malloc(size * sizeof(char));

	if(*arr == NULL) {
		return 1;
	}

	if(fillRandom) {
		srand(NULL);
		for(int i = 0; i < size; i++) {
			(*arr)[i] = (char)(rand() % 5 + 'A');
		}
		(*arr)[size - 1] = (char)(((*arr)[size - 2] + 32) % 256);

	} else {
		for(int i = 0; i < size; i++) {
			(*arr)[i] = 'a';
		}

		return 0;
	}
	return 0;
}

void checkOutput(char *input, char *outText, int *outAmount, int outputSize)
{
	bool outputOk = true;
	long errorCount = 0;
	for(int outI = 0, inI = 0; outI < outputSize && inI < INPUT_SIZE; outI++)
	{
		if(input[inI] == outText[outI]) {
			int count = inI + outAmount[outI];
			while(inI < count) {
				//printf("Input: %c, Output: %c, Amount: %d\n", input[inI], outText[outI], outAmount[outI]);
				if(input[inI] != outText[outI]) {
					printf("Error at %d in outAmount\n", outI);
					errorCount++;
					outputOk = false;
				}
				inI++;
			}
		} else {
			printf("Error at %d in outText\n", outI);
			errorCount++;
			outputOk = false;
		}
	}
	if(outputOk) {
		printf("Output checked: no errors\n");
	} else {
		printf("Output checked: total %l errors\n", errorCount);
	}
}

int main()
{
	char *in = 0;
	char *outText = 0;
	int *outAmount = 0;


	initializeCharArray(&in, INPUT_SIZE, true);
	
	initializeArray(&outAmount, INPUT_SIZE, false);
	initializeCharArray(&outText, INPUT_SIZE, false);
	double start = (double)clock();

	char current = in[0];
	int amount = 0, outI = 0;
	outText[0] = current;
	for(int inI = 0; inI < INPUT_SIZE; inI++) {
		if(in[inI] != current) {
			outAmount[outI] = amount;
			outI++;
			outText[outI] = current = in[inI];
			amount = 1;
		} else {
			amount++;
		}
	}

	double end = (double)clock();

	start /= CLOCKS_PER_SEC;
	end /= CLOCKS_PER_SEC;
	printf("Cpu runtime: %f\n", (end - start) * 1000);
	checkOutput(in, outText, outAmount, outI + 1);
	getchar();
	return 0;
}