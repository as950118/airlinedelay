#include <iostream>
#include <string>
using namespace std;
#define debug 0
int n;
string S;
string engs = "aiylneortuvw";
string lat[12] = {"as", "ios", "ios", "les", "anes", "anes", "os", "res", "tas", "us", "ves", "was"};
string eng_last;
int idx_last;

string if_not_in_engs(string s){
	string append_lat;

	append_lat = "us";
	s.append(append_lat);
	return s;
}

string if_in_engs(string s){
	string append_lat;

	s.erase(s.size()-1, s.size());
	append_lat = lat[idx_last];
	s.append(append_lat);
	return s;
}

string if_eng_is_e(string s){
	string if_n;
	string append_lat;
	
	if_n = s[s.size() -2];
	if(s.size()>1 && if_n.compare("n") == 0){
		s.erase(s.size()-2, s.size());
		append_lat = lat[idx_last];
		s.append(append_lat);
	}
	else{
		append_lat = "us";
		s.append(append_lat);
	}
	return s;
}

string func(string s){
	
	eng_last = s[s.size() -1];
	idx_last = engs.find(eng_last);

	if(engs.find(eng_last) == string::npos){
		return if_not_in_engs(s);
	}

	if(idx_last == 5){
		return if_eng_is_e(s);
	}
	return if_in_engs(s);
}

int main() {
	cin>>n;
	for(int i=0; i<n; i++){
		cin>>S;
		cout<<func(S)<<endl;
		S.clear();		
	}
	return 0;
}
