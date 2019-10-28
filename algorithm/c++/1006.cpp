#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace std;

int n, w; 
int map[10000][2];
int cache[10000][4][4];

// 0 -> not used, 1 -> inner used, 2 -> outer used, 3 -> both used
// prev -> ���� ������ ����, last -> n-1�� ���� ����
int func(int index, int prev, int last)
{
	int& ret = cache[index][prev][last];
	
	//�̹� �湮�ߴٸ� ���� 
	if (ret){
		return ret;
	}
	
	//�ΰ��� �������� ����
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

	// index�� ���� �������� ��
	if (index == n - 1) {
		//�׸��� ���� n-1==0 �� ���� ���� �ƴ϶� 1ĭ�� ��� 
		if (index == 0){
			return both ? 1 : 2;
		}
		ret = 2;
		//���� ����� ������ ������ ��� 
		if (last == 0) {
			//inner �����ϰ� 
			if(inner){
				//������ outer ������� ��� 
				if(prev==0 || prev==2){
					ret = 1;
				}
			}
			//outer�� �����ϰ� 
			if(outer){
			//������ 
				if(prev==0 || prev==1){
					ret = 1;
				}
			}
			//both�� �����Ұ��
			if(both){
				ret= 1;
			}
		}
		
		//���� ����� inner�� ������� ��� 
		if (last == 1) {
			//outer�� �����ϰ� 
			if(outer){
			//������ 0�̳� 1�̸� 
				if(prev==0 || prev==1){
					ret = 1;
				}
			}
		}
		
		//���� ����� outer�� ������� ��� 
		if (last == 2) {
			//inner �����ϰ� 
			if(inner){
				//������ Ȧ��, �� �Ʒ���� 
				if(prev==0 || prev==2){
					ret = 1;
				}
			}
		}
		
		return ret;
	}
	
	// ���� �ϳ��� ���������� 
	ret = 2 + func(index + 1, 0, index ? last : 0);
	// inner ����
	if (inner && (prev==0 || prev==2)){
		ret = min(ret, 1 + func(index + 1, 1, index ? last : 1));
	}
	// outer ���� 
	if (outer && (prev ==0 || prev==1)){
		ret = min(ret, 1 + func(index + 1, 2, index ? last : 2));
	} 
	// inner outer �� ���� 
	if (inner && outer && prev == 0){
		ret = min(ret, func(index + 1, 3, index ? last : 3));
	} 
	// both ���� 
	if (both){
		ret = min(ret, 1 + func(index + 1, 3, index ? last : 0));
	}
	
	return ret;
}
int main()
{
	//�׽�Ʈ ���̽� ���� 
	int t;
	cin>>t;
	while (t--) {
		//�ʱ�ȭ 
		memset(map, 0, sizeof(map));
		memset(cache, 0, sizeof(cache));
		
		//�Է� 
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
