// Source: https://usaco.guide/general/io

#include <bits/stdc++.h>
#include <cstdio>
using namespace std;

int main() {
	freopen("angry.in", "r", stdin);
	freopen("angry.out", "w", stdout);
	int n,k;
	
	cin>>n>>k;
	vector<int> haybales(n);

	for(int i=0;i<n;i++){
		cin>>haybales[i];
	}
	sort(haybales.begin(),haybales.end());
	int lo = 1, hi = 1e9;
	int sol = 1e9;

	while(lo<=hi){
		int mid = lo + (hi - lo) / 2;
		int i = 0;
		for (int kas = 0; kas < k && i < n; kas++) {
			int hasta = haybales[i];
			hasta += mid * 2;
			while (i < n - 1 && haybales[i + 1] <= hasta) i++;
			i++;
		}

		bool ok;
		ok = (i >= n);
		if (ok) {
			sol = min(sol, mid);
			hi = mid - 1;
		} else {
			lo = mid + 1;
		}
	}
	cout<<sol;
}
