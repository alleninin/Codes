#include "testlib.h"
#include <bits/stdc++.h>

using namespace std;

int main(int argc, char* argv[]) {
	registerValidation(argc, argv);
	
	int n = inf.readInt(1, 200 * 1000, "n");
	inf.readSpace();
	inf.readInt(0, n, "k");
	inf.readEoln();
	inf.readInts(n, 1, 1000 * 1000 * 1000, "a");
	inf.readEoln();
	inf.readEof();
}
