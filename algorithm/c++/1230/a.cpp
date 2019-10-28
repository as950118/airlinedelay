#include <iostream>
using namespace std;
int a[4] = {0, };
int sum = 0;
int temp;
int func(){
	
	for(int i=0; i<4; i++){
		if(sum == a[i]*2){
			return 1;
		}
		for(int j=i+1; j<4; j++){
			if(sum == 2*(a[i]+a[j])){
				return 1;
			}
		}
	}
	return 0;
}
int main() {
	for(int i =0; i<4; i++){
		cin>>temp;
		sum += temp;
		a[i] = temp;
	}
	if(func()){
		cout<<"YES";
	}
	else{
		cout<<"NO";
	}
}
