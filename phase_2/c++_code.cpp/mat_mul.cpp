#include<bits/stdc++.h>
using namespace std;
vector<double>mat_mul(vector<double>&A, vector<double>&B,int ra,int ca,int rb,int cb){
    if(ca!=rb){
        cout<<"Matrix multiplication not possible"<<endl;
        return {};
    }
    vector<double>c(ra*cb,0);
    for(int i=0;i<ra;i++){
        for(int j=0;j<cb;j++){
            double ans=0;
            for(int k=0;k<ca;k++){
                ans +=A[i*ca+k]*B[k*cb+j];
            }
            c[i*cb+j]=ans;
        }
    }
    return c;
}
int main() {
    int ra, ca, rb, cb;

    cin >> ra >> ca;
    vector<double> A(ra * ca);
    for (double &x : A) cin >> x;

    cin >> rb >> cb;
    vector<double> B(rb * cb);
    for (double &x : B) cin >> x;

    vector<double> C = mat_mul(A, B, ra, ca, rb, cb);

    if (C.empty()) return 0;

    cout << "Result:\n";
    for (int i = 0; i < ra; i++) {
        for (int j = 0; j < cb; j++) {
            cout << C[i * cb + j] << " ";
        }
        cout << '\n';
    }
}