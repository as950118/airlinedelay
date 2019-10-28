#include <iostream>
#define MIN(a,b) (a<b ? a:b)
using namespace std;


int func(){
	int ret=0;
	int c,m,x;
	cin>>c;
	cin>>m;
	cin>>x;
	ret += MIN(MIN(c,m), x);
	c -= MIN(MIN(c,m), x);
	m -= MIN(MIN(c,m), x);
	ret += MIN(MIN(c,m), (c+m)/3);
	//ret += ((x<c ? x:c)<m ? (x<c ? x:c) : m);
	//c -= ((x<c ? x:c)<m ? (x<c ? x:c) : m);
	//m -= ((x<c ? x:c)<m ? (x<c ? x:c) : m);
	//ret += ( (c+m)/3 < (c<m?c:m) ? (c+m)/3 : (c<m?c:m) );
	return ret;
}

int main() {
	int q;
	int ret;
	cin>>q;
	while(q--){
		ret = func();
		cout<<ret<<endl;
	}
	return 0;
}
