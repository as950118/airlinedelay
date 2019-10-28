#include<iostream>
#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<queue>
#include<vector>
#include<stack>
#define ll long long

using namespace std;

const ll MAXN=2005;

vector<int>v[MAXN];
int day[MAXN], per[MAXN], st[MAXN];
//st == 그날에 작업을 시작해야만 하는 사람들의 수 
struct Point
{
    int id,cnt,next;
    Point(int i,int c,int n){id=i;cnt=c;next=n;}
    bool operator < (const Point& r)const
    {
        return cnt<r.cnt;
    }
};
int main()
{
    int m,n,w,h;
    priority_queue<Point>q;
    queue<Point>q1;

    cin>>m>>n>>w>>h;

    for(int i=1;i<=m;i++){
        cin>>day[i];
    }
    
    st[0]=0;
    
	for(int i=1;i<=n;i++){
        cin>>per[i];
        st[i]=per[i];
    }
    
    for(int i=1;i<=n;i++){
        for(int j=i+1;j<=i+w-1;j++){
        	cout<<j<<" "<<i<<endl;
            cout<<st[j]<<" "<<st[i]<<endl;
            st[j]-=st[i];    
		    for(int k=1;k<=n;k++){
		    	cout<<st[k]<<" ";
		    }
		    cout<<endl;
        }
    }
    
    for(int i=1;i<=n;i++){
    	cout<<st[i]<<" ";
    }
    //return 1;
    bool f=1;
    for(int i=1;i<=m;i++){
        q.push(Point(i,day[i],1));
    }
    
    for(int i=1;i<=n;i++){
        while(q1.size()&&q1.front().next==i){
            Point now=q1.front();
            q1.pop();
            q.push(now);
        }
        while(st[i]--){
            if(q.size()==0){f=0;break;}
            Point now=q.top();
            q.pop();
            if(i+w-1>n){f=0;break;}
            v[now.id].push_back(i);
            now.next=i+w+h;
            now.cnt-=w;
            if(now.cnt!=0)q1.push(now);
        }
    }
    
    if(q.size()||q1.size())f=0;
    if(f)cout<<1<<endl;
    else {
        cout<<-1<<endl;return 0;
    }
    
    for(int i=1;i<=m;i++){
        int len=v[i].size();
        for(int j=0;j<len;j++){
            printf("%d%c",v[i][j],(j==len-1)?'\n':' ');
        }
    }
}
