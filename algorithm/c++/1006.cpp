#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace std;

int n, w; 
int map[10000][2];
int cache[10000][4][4];

// 0 -> not used, 1 -> inner used, 2 -> outer used, 3 -> both used
// prev -> 이전 구역의 상태, last -> n-1번 블럭의 상태
int func(int index, int prev, int last)
{
	int& ret = cache[index][prev][last];
	
	//이미 방문했다면 종료 
	if (ret){
		return ret;
	}
	
	//두개가 가능한지 여부
	int prev_index; 
	if(index){
		prev_index = index-1;
	}
	else{
		prev_index = n-1;
	}
	bool inner = (map[index][0] + map[prev_index][0] <= w);
	bool outer = (map[index][1] + map[prev_index][1] <= w);
	bool both = (map[index][0] + map[index][1] <= w);

	// index가 끝에 도달했을 때
	if (index == n - 1) {
		//그리고 만약 n-1==0 인 경우는 원이 아니라 1칸인 경우 
		if (index == 0){
			return both ? 1 : 2;
		}
		ret = 2;
		//이전 블록이 사용안한 상태일 경우 
		if (last == 0) {
			//inner 가능하고 
			if(inner){
				//이전이 outer 사용했을 경우 
				if(prev==0 || prev==2){
					ret = 1;
				}
			}
			//outer가 가능하고 
			if(outer){
			//이전이 
				if(prev==0 || prev==1){
					ret = 1;
				}
			}
			//both가 가능할경우
			if(both){
				ret= 1;
			}
		}
		
		//이전 블록이 inner만 사용했을 경우 
		if (last == 1) {
			//outer가 가능하고 
			if(outer){
			//이전이 0이나 1이면 
				if(prev==0 || prev==1){
					ret = 1;
				}
			}
		}
		
		//이전 블록이 outer만 사용했을 경우 
		if (last == 2) {
			//inner 가능하고 
			if(inner){
				//이전이 홀수, 즉 아래라면 
				if(prev==0 || prev==2){
					ret = 1;
				}
			}
		}
		
		return ret;
	}
	
	// 각각 하나씩 배정했을때 
	ret = 2 + func(index + 1, 0, index ? last : 0);
	// inner 가능
	if (inner && (prev==0 || prev==2)){
		ret = min(ret, 1 + func(index + 1, 1, index ? last : 1));
	}
	// outer 가능 
	if (outer && (prev ==0 || prev==1)){
		ret = min(ret, 1 + func(index + 1, 2, index ? last : 2));
	} 
	// inner outer 다 가능 
	if (inner && outer && prev == 0){
		ret = min(ret, func(index + 1, 3, index ? last : 3));
	} 
	// both 가능 
	if (both){
		ret = min(ret, 1 + func(index + 1, 3, index ? last : 0));
	}
	
	return ret;
}
int main()
{
	//테스트 케이스 숫자 
	int t;
	cin>>t;
	while (t--) {
		//초기화 
		memset(map, 0, sizeof(map));
		memset(cache, 0, sizeof(cache));
		
		//입력 
		cin>>n;
		cin>>w;
		for (int i = 0; i<n; i++){
			cin>>map[i][0];
		}
		for (int i = 0; i<n; i++){
			cin>>map[i][1];
		}
		cout<<func(0, 0, 0)<<endl;
	}
	return 0;
}
