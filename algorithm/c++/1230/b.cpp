#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <algorithm>
using namespace std;
int n,k;
string s;
int ret;
/*
void func(){
	int temp;
	for(int i=0; i<n; i++){
		ret[i] = (s/pow(10, i));
		ret[i] %= 10;
	}
	return;
}
*/

int main() {
	cin>>n>>k>>s;
	//func();
	char temp = s.at(0);
	char com[] = "1";
	if(k && n==1){
		ret = 0;
		k-=1;
	}
	else if(k && strcmp(&temp, com) != 0){
		ret = 1;
		k-=1;
	}
	else{
		ret = temp;
	}
	cout<<ret;
	
	int i;
	char com2[] = "0";
	
	for(i=1; i<n && k; i++){
		temp = s.at(i);
		if(strcmp(&temp, com2) != 0){
			k--;
		}
		cout<<0;
	}
	for(i; i<n; i++){
		cout<<s[i];
	}
	return 0;
}
