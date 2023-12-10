#ifndef InstantGNN_H
#define InstantGNN_H
#include<iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <string>
#include <unistd.h>
#include <sys/time.h>
#include<Eigen/Dense>
#include "SpeedPPR.h"
#include "Graph.h"

using namespace std;
using namespace Eigen;
typedef unsigned int uint;
namespace propagation{
    class Instantgnn{
        
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int NUMTHREAD=40;//Number of threads
        uint edges, vert;
        Graph g;
        Eigen::MatrixXd X;
        SpeedPPR ppr;
        vector<vector<double>> R;
        vector<vector<double>> R_b;
        vector<vector<double>> pi_b;
        vector<vector<double>> inaccaracy_pos;
        vector<vector<double>> inaccaracy_neg;
    	float rmax,alpha,t,epsilon,lower_threshold,omega;
        int rw_count=0;
        unsigned long long num_total_rw;
        string dataset_name;
        string updateFile;
        vector<double>rowsum_pos;
        vector<double>rowsum_neg;
        vector<int>random_w;
        vector<int>update_w;
        vector<double>Du;
        Config config;
        int dimension;
        double initial_operation(string path, string dataset,uint mm,uint nn,double rmaxx,double rbmax, double delta, double alphaa,double epsilonn,Eigen::Map<Eigen::MatrixXd> &feat, string algorithm);
        void ppr_push(int dimension, Eigen::Ref<Eigen::MatrixXd>feat, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, bool log, string algorithm, bool reverse);
        void ppr_residue(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, string algorithm, bool reverse);
        void snapshot_operation(string updatefilename, double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat, string algorithm);
        void snapshot_lazy(string updatefilename, double rmaxx,double rbmax, double delta,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat, Eigen::Map<Eigen::MatrixXd> &feat_p, Eigen::Map<Eigen::MatrixXd> &change_node_list, string algorithm);
        vector<vector<uint>> update_graph(string updatefilename, vector<uint>&affected_nodelst, vector<vector<uint>>&delete_neighbors);
    };
}


#endif // InstantGNN_H