#include <iostream>
using namespace std;

int func(){
	int n=0;
	int ret=0;
	int tmp=0;
	cin>>n;
	while(n--){
		cin>>tmp;
		if(tmp==2048)
			//cout<<"YES"<<endl;
			return 1;
		else if(tmp>2048)
			continue;
		else
			ret += tmp;
		if(ret>=2048)
			//cout<<"YES"<<endl;
			return 1;
	}
	return 0;
}

int main() {
	int q;
	int ret;
	cin>>q;
	while(q--){
		ret = func();
		if(ret)
			cout<<"YES"<<endl;
		else
			cout<<"NO"<<endl;
	}
	return 0;
}
