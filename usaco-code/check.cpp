#include "testlib.h"
#include <bits/stdc++.h>

using namespace std;

int n, k;
vector<int> a;

int readAnswer(InStream &in) {
	int ans = in.readInt(-1, 1000000000);

	if (ans == -1)
		return ans;
	else if (ans == 0)
	    in.quitf(_pe, "Unexpected 0");
		
	int cnt = 0;
	for (int i = 0; i < n; ++i)
		if (a[i] <= ans)
		    cnt++;
	
	if (cnt != k)
		in.quitf(_wa, "The number of less or equal elements must be equal to %d but it equals to %d", k, cnt);
		
	return ans;
}

int main(int argc, char* argv[]) {
	registerTestlibCmd(argc, argv);
	
	n = inf.readInt();
	k = inf.readInt();
	
	a = vector<int>(n);
	for (int i = 0; i < n; ++i)
		a[i] = inf.readInt();
		
	int jans = readAnswer(ans);
	int pans = readAnswer(ouf);
	
	if (jans == -1) {
		if (pans != -1)
			quitf(_fail, "Participant found answer but jury doesn't");
	} else {
		if (pans == -1)
			quitf(_wa, "Jury found answer but participant doesn't");
	}
	
	quitf(_ok, "%d", pans);
}
