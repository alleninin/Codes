// Source: https://usaco.guide/general/io

#include <bits/stdc++.h>
using namespace std;

int main() {
	
	int t;cin>>t;
	for(int asdfsdf=0;asdfsdf<t;++asdfsdf){
		int n,k,g;cin>>n>>k>>g;
		int totalsilver=k*g;
		int totalspent=0;
		int a;
		if(g%2==0){
			 a=(ceil(g/2)-1)*n;}
		else{
			 a=(g/2)*n;}
		vector<int> asdf(n,ceil(g/2)-1);
		if(a>totalsilver){
			cout<<k*g<<"\n";
			continue;
		}
		
		else if((totalsilver-a)%g==0){
			cout<<(ceil(g/2)-1)*n<<"\n";
			
			continue;
		}
		else if(((totalsilver-a)>g)){
			double aa=((totalsilver-a+g)/g)*g;
			cout<<totalsilver-aa<<"\n";
			continue;
		}
		else if(((totalsilver-a)<g)){
			int kk=asdf[0]+totalsilver-a;

			cout<<a-asdf[0]-((g-(kk%g)));
		}
		

	cout<<"\n";
	}
}