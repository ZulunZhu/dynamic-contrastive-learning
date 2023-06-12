#include "instantAlg.h"
#include "Graph.h"

using namespace std;
using namespace Eigen;

namespace propagation
{
vector<vector<uint>> Instantgnn::update_graph(string updatefilename, vector<uint>&affected_nodelst, vector<vector<uint>>&delete_neighbors) // vector<vector<uint>>&add_adjs
{
    ifstream infile(updatefilename.c_str());
    //cout<<"updating graph " << updatefilename <<endl;
    uint v_from, v_to;
    int insertFLAG = 0;

    vector<vector<uint>> new_neighbors(vert);
    vector<bool> isAffected(vert, false);
    while (infile >> v_from >> v_to)
    {
        insertFLAG = g.isEdgeExist(v_from, v_to);
        
        // update graph
        if(!isAffected[v_from]){
            affected_nodelst.push_back(v_from);
            isAffected[v_from] = true;
        }
        
        if(insertFLAG == 1){
            g.insertEdge(v_from, v_to);
            new_neighbors[v_from].push_back(v_to);
        }
        else if(insertFLAG == -1){
            cout<<"delete......"<<endl;
            g.deleteEdge(v_from, v_to);
            delete_neighbors[v_from].push_back(v_to);
        }
    }
    infile.close();

    cout<<"update graph finish..."<<"affected_nodelst.size():"<<affected_nodelst.size()<<endl;

    return new_neighbors;
}

bool err_cmp_pos(const pair<int,double> a,const pair<int,double> b){
	return a.second > b.second;
}
bool err_cmp_neg(const pair<int,double> a,const pair<int,double> b){
	return a.second < b.second;
}
//batch_lazy_update
void Instantgnn::snapshot_lazy(string updatefilename, double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat,Eigen::Map<Eigen::MatrixXd> &feat_p, Eigen::Map<Eigen::MatrixXd> &change_node_list, string algorithm)
{
    alpha=alphaa;
    rmax=rmaxx;

    vector<queue<uint>> candidate_sets(dimension);
    vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));

    vector<queue<uint>> candidate_sets_reverse(dimension);
    vector<vector<bool>> isCandidates_reverse(dimension, vector<bool>(vert, false));
    vector<bool> isUpdateW(dimension, false);

    clock_t start_t, end_t;
    start_t = clock();
    cout<<"Lazy updating begin, for snapshot: " << updatefilename <<endl;
    
    //update graph, obtain affected node_list
    vector<uint> affected_nodelst;

    //reverse push, get the affected node_list really needed to push

    vector<uint> changed_nodes;
    vector<vector<uint>> delete_neighbors(vert);
    vector<vector<uint>> add_neighbors(vert);

    add_neighbors = update_graph(updatefilename, affected_nodelst, delete_neighbors);
    end_t = clock();
    //cout<<"-----update_graph finish-------- time: " << (end_t - start_t)/(1.0*CLOCKS_PER_SEC)<<" s"<<endl;
    //cout<<"affected_nodelst.size():"<<affected_nodelst.size()<<endl;

    //deal nodes in affected node_list, update \pi and r
    vector<double> oldDu(affected_nodelst.size(), 0);
    //double oldDu[affected_nodelst.size()];


    // cout<<'------------------ reverse push -------------------'<<endl;
    double n = feat.rows();
    // cout<<"****number of nodes"<<n <<endl;

    double errorlimit=1.0/n;
    double epsrate=1;
    config.rbmax = errorlimit*epsrate;
    
    
    

    for(uint i=0;i<affected_nodelst.size();i++)
    {
        uint affected_node = affected_nodelst[i];
        // update Du
        oldDu[i] = Du[affected_node]; //[d(u)-delta_d(u)]^0.5
        Du[affected_node] = pow(g.getOutSize(affected_node), 0.5);
        
        //update \pi(u) to avoid dealing with N(u), r needs to be updated accordingly
        for(int dim=0; dim<dimension; dim++)
        {
            feat(affected_node,dim) = feat(affected_node,dim) * Du[affected_node] / oldDu[i];
            double delta_1 = feat(affected_node,dim) * (oldDu[i]-Du[affected_node]) / alpha / Du[affected_node];
            R[dim][affected_node] += delta_1;
        }
    }
    // MSG(feat(36,36));
    clock_t end_t2 = clock();
    //cout << "-----update pi and r finish----- time: "<< (end_t2 - end_t)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;
    
    // *******************Perform the reverse push***********************//
    R_b = vector<vector<double>>(dimension, vector<double>(vert, 0));
    pi_b = vector<vector<double>>(dimension, vector<double>(vert, 0));
    cout<<" Performing reverse push..."<<endl;
    for(uint i=0; i<affected_nodelst.size(); i++)
    {
        uint affected_node = affected_nodelst[i];
        for(int dim=0; dim<dimension; dim++)
        {   
            R_b[dim][affected_node] = feat(affected_node, dim);
            candidate_sets_reverse[dim].push(affected_node);
            isCandidates_reverse[dim][affected_node] = true;

        }    
    }
    Instantgnn::ppr_push(dimension, feat, true,candidate_sets_reverse,isCandidates_reverse,true,algorithm, true);

    inaccaracy_pos = vector<vector<double>>(dimension, vector<double>(vert, 1));
    inaccaracy_neg = vector<vector<double>>(dimension, vector<double>(vert, -1));
    
    // update the inaccuracy
    for(int i=0; i<dimension; i++)
    {   
        double test = 0;
        double rsum_pos=0;
	    double errsum_pos=0;
        double rsum_neg=0;
	    double errsum_neg=0;
        vector< pair<int,double> > error_pos_idx;
        vector< pair<int,double> > error_neg_idx;
        for(uint j=0; j<vert; j++)
        {   
            double pmin;
            double r_mul_inacc;
            vector< pair<int,double> > error_idx;
            if(pi_b[i][j]>0){
                pmin=min((pi_b[i][j]+config.rbmax_p)*(1-alpha)/alpha,1.0);
                inaccaracy_pos[i][j]*=(1-pmin/g.getOutSize(j));

            }else{
                pmin=max((pi_b[i][j]+config.rbmax_n)*(1-alpha)/alpha,-1.0);
                inaccaracy_neg[i][j]*=(1+pmin/g.getOutSize(j));
            }

            if((1-inaccaracy_pos[i][j])==0){
                r_mul_inacc = R[i][j]*(-1-inaccaracy_pos[i][j]);

            }else if((-1-inaccaracy_neg[i][j]==0)){
                r_mul_inacc = R[i][j]*(1-inaccaracy_pos[i][j]);
            }else{
                cout<<" Fault with the residue*error value!"<<endl;
                exit(0);
            }
            

            if(r_mul_inacc>0){
                errsum_pos+=r_mul_inacc;
                error_pos_idx.push_back(make_pair(j,r_mul_inacc));
            }else{
                errsum_neg+=r_mul_inacc;
                error_neg_idx.push_back(make_pair(j,r_mul_inacc));
            }
            if(R[i][j]>0){
                rsum_pos+=R[i][j];
            }else{
                rsum_neg+=R[i][j];
            }
            
            // if(inaccaracy_neg[i][j]==-0.8){
            //     MSG(alpha);
            //     MSG(g.getOutSize(j));
            //     MSG(config.rbmax_n);
            //     MSG(rowsum_neg[i]);
            //     cout<<"pi_b"<<i<<"|"<<j<<"::"<<pi_b[i][j]<<endl;
            //     cout<<"inaccaracy"<<i<<"|"<<j<<"::"<<inaccaracy_neg[i][j]<<endl;

            // }
        }

        double errbound_pos=rsum_pos*0.05;
        double errbound_neg=rsum_neg*0.05;
        sort(error_pos_idx.begin(), error_pos_idx.end(), err_cmp_pos);
        sort(error_neg_idx.begin(), error_neg_idx.end(), err_cmp_neg);
        long rank = 0;
        while(errsum_pos>errbound_pos){

            uint current_node = error_pos_idx[rank].first;
            if(change_node_list(current_node)==0){
                change_node_list(current_node) = 1;
                inaccaracy_pos[i][current_node] = 1;
                changed_nodes.push_back(current_node);
                
            }
            errsum_pos-=error_pos_idx[rank].second;
            rank++;
        }
        
        rank = 0;
        while(errsum_neg<errbound_neg){
            //cout<<i<<" : "<<errsum<<"--"<<error_idx[i].second<<endl;;
            uint current_node = error_neg_idx[rank].first;
            if(change_node_list(current_node)==0){
                change_node_list(current_node) = 1;
                inaccaracy_neg[i][current_node] = -1;
                changed_nodes.push_back(current_node);
            }
            errsum_neg-=error_neg_idx[rank].second;
            rank++;
        }
        
        
    }
    MSG(affected_nodelst.size());
    MSG(changed_nodes.size());
    
    
    
   // *******************Update the embedding***********************//


    //update r
    for(uint i=0; i<affected_nodelst.size(); i++)
    {
        uint affected_node = affected_nodelst[i];
        for(int dim=0; dim<dimension; dim++)
        {
            double rowsum_p=rowsum_pos[dim];
            double rowsum_n=rowsum_neg[dim];
            double rmax_p=rowsum_p*rmax;
            double rmax_n=rowsum_n*rmax;
            
            double increment = feat(affected_node,dim) + alpha*R[dim][affected_node] - alpha*X(affected_node,dim);
            increment *= oldDu[i] - Du[affected_node];
            increment /= Du[affected_node];
            
            for(uint j=0; j<add_neighbors[affected_node].size(); j++)
            {
                uint add_node = add_neighbors[affected_node][j];
                increment += (1-alpha)*feat(add_node,dim) / Du[affected_node] / Du[add_node];
            }
            for(uint j=0; j<delete_neighbors[affected_node].size(); j++)
            {
                uint delete_node = delete_neighbors[affected_node][j];
                increment -= (1-alpha)*feat(delete_node,dim) / Du[affected_node] / Du[delete_node];
            }
            increment /= alpha;
            R[dim][affected_node] += increment;
            
            if( R[dim][affected_node]>rmax_p || R[dim][affected_node]<rmax_n )
            {
                if(!isCandidates[dim][affected_node]){
                    candidate_sets[dim].push(affected_node);
                    isCandidates[dim][affected_node] = true;
                }
                if(!isUpdateW[dim]){
                    update_w.push_back(dim);
                    isUpdateW[dim] = true;
                }
            }
        }
    }
    clock_t end_t3 = clock();
    //cout<<"-----update r finish----- time: "<<(end_t3 - end_t2)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;
  
    //push
    if(update_w.size()>0)
    {
      cout<<"dims of feats that need push:"<<update_w.size()<<endl;
      if(algorithm == "instant"){
        cout<<"before push feat(36,36):"<<feat(36,36)<<endl;
        Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm,false);
        cout<<"after push feat(36,36):"<<feat(36,36)<<endl;
      }
      else if(algorithm == "speed_push"){
        cout<<"before push feat(36,36):"<<feat(36,36)<<endl;
        Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm,false);
        cout<<"after push feat(36,36):"<<feat(36,36)<<endl;
        cout<<"speed push"<<endl;
      }
    }

    // **************************Assign the positive sample****************//
    for(uint i=0; i<changed_nodes.size(); i++)
    {
        uint changed_node = changed_nodes[i];
        // MSG(change_node_list(changed_node,1));
        for(int dim=0; dim<dimension; dim++)
        {
            feat_p(changed_node,dim) = feat(changed_node,dim);
        }
        
    }    
    

}




//batch_update
void Instantgnn::snapshot_operation(string updatefilename, double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat, string algorithm)
{
    alpha=alphaa;
    rmax=rmaxx;

    vector<queue<uint>> candidate_sets(dimension);
    vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));
    vector<bool> isUpdateW(dimension, false);

    clock_t start_t, end_t;
    start_t = clock();
    cout<<"updating begin, for snapshot: " << updatefilename <<endl;
    
    //update graph, obtain affected node_list
    vector<uint> affected_nodelst;

    vector<vector<uint>> delete_neighbors(vert);
    vector<vector<uint>> add_neighbors(vert);

    add_neighbors = update_graph(updatefilename, affected_nodelst, delete_neighbors);
    end_t = clock();
    //cout<<"-----update_graph finish-------- time: " << (end_t - start_t)/(1.0*CLOCKS_PER_SEC)<<" s"<<endl;
    //cout<<"affected_nodelst.size():"<<affected_nodelst.size()<<endl;

    //deal nodes in affected node_list, update \pi and r
    vector<double> oldDu(affected_nodelst.size(), 0);
    //double oldDu[affected_nodelst.size()];

    
    for(uint i=0;i<affected_nodelst.size();i++)
    {
        uint affected_node = affected_nodelst[i];
        // update Du
        oldDu[i] = Du[affected_node]; //[d(u)-delta_d(u)]^0.5
        Du[affected_node] = pow(g.getOutSize(affected_node), 0.5);
        
        //update \pi(u) to avoid dealing with N(u), r needs to be updated accordingly
        for(int dim=0; dim<dimension; dim++)
        {
            feat(affected_node,dim) = feat(affected_node,dim) * Du[affected_node] / oldDu[i];
            double delta_1 = feat(affected_node,dim) * (oldDu[i]-Du[affected_node]) / alpha / Du[affected_node];
            R[dim][affected_node] += delta_1;
        }
    }
    // MSG(feat(36,36));
    clock_t end_t2 = clock();
    //cout << "-----update pi and r finish----- time: "<< (end_t2 - end_t)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;
  
    //update r
    for(uint i=0; i<affected_nodelst.size(); i++)
    {
        uint affected_node = affected_nodelst[i];
        for(int dim=0; dim<dimension; dim++)
        {
            double rowsum_p=rowsum_pos[dim];
            double rowsum_n=rowsum_neg[dim];
            double rmax_p=rowsum_p*rmax;
            double rmax_n=rowsum_n*rmax;
            
            double increment = feat(affected_node,dim) + alpha*R[dim][affected_node] - alpha*X(affected_node,dim);
            increment *= oldDu[i] - Du[affected_node];
            increment /= Du[affected_node];
            
            for(uint j=0; j<add_neighbors[affected_node].size(); j++)
            {
                uint add_node = add_neighbors[affected_node][j];
                increment += (1-alpha)*feat(add_node,dim) / Du[affected_node] / Du[add_node];
            }
            for(uint j=0; j<delete_neighbors[affected_node].size(); j++)
            {
                uint delete_node = delete_neighbors[affected_node][j];
                increment -= (1-alpha)*feat(delete_node,dim) / Du[affected_node] / Du[delete_node];
            }
            increment /= alpha;
            R[dim][affected_node] += increment;
            
            if( R[dim][affected_node]>rmax_p || R[dim][affected_node]<rmax_n )
            {
                if(!isCandidates[dim][affected_node]){
                    candidate_sets[dim].push(affected_node);
                    isCandidates[dim][affected_node] = true;
                }
                if(!isUpdateW[dim]){
                    update_w.push_back(dim);
                    isUpdateW[dim] = true;
                }
            }
        }
    }
    clock_t end_t3 = clock();
    //cout<<"-----update r finish----- time: "<<(end_t3 - end_t2)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;

    //push
    if(update_w.size()>0)
    {
      cout<<"dims of feats that need push:"<<update_w.size()<<endl;
      if(algorithm == "instant"){
        cout<<"before push feat(36,36):"<<feat(36,36)<<endl;
        Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm,false);
        cout<<"after push feat(36,36):"<<feat(36,36)<<endl;
      }
      else if(algorithm == "speed_push"){
        cout<<"before push feat(36,36):"<<feat(36,36)<<endl;
        Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm,false);
        cout<<"after push feat(36,36):"<<feat(36,36)<<endl;
        cout<<"speed push"<<endl;
      }

      
    }
}

int startsWith(string s, string sub){
        return s.find(sub)==0?1:0;
}
double Instantgnn::initial_operation(string path, string dataset,uint mm,uint nn,double rmaxx,double alphaa,double epsilonn, Eigen::Map<Eigen::MatrixXd> &feat, string algorithm)
{   
    // ppr.just_fortest();
    if(algorithm=="instant"){
        X = feat; // change in feat not influence X
    }
    

    // dimension=feat.cols();
    // cout<<"dimension: "<<dimension<<", col:"<<feat.rows()<<endl;

    dimension=min(feat.rows(),feat.cols());
    cout<<"dimension: "<<dimension<<", col:"<<max(feat.rows(),feat.cols())<<endl;
    
    rmax=rmaxx;
    edges=mm;
    vert=nn;
    alpha=alphaa;
    epsilon=epsilonn;
    omega = (2+epsilon)*log(2*vert)*vert/epsilon/epsilon;
    dataset_name=dataset;
    cout<<dataset_name<<endl;
    if(algorithm == "instant"){
        cout<<"Using Instant!"<<endl;
        
        g.inputGraph(path, dataset_name, vert, edges);
        // g.inputGraph_fromedgelist(path, dataset_name, vert, edges);

    }
    else if(algorithm == "speed_push"){
        cout<<"Using SpeedPPR!"<<endl;
        g.inputGraph_fromedgelist(path, dataset_name, vert, edges);
        // if (!bin_file.good()) {
        //     CleanGraph cleaner;
        //     cleaner.clean_graph(dataset_name,path);
        // }
        g.read_for_speedppr(path, dataset_name, vert, edges);
        ppr.load_graph(g,alpha);
        // ppr.graph.set_dummy_neighbor(ppr.graph.get_dummy_id());
        ppr.graph.reset_set_dummy_neighbor();
        ppr.graph.fill_dead_end_neighbor_with_id();
        lower_threshold=1.0 / vert;

    }
    
    
    
    Du=vector<double>(vert,0);
    double rrr=0.5;
    for(uint i=0; i<vert; i++)
    {
        Du[i]=pow(g.getOutSize(i),rrr);
    }

    R = vector<vector<double>>(dimension, vector<double>(vert, 0));
    rowsum_pos = vector<double>(dimension,0);
    rowsum_neg = vector<double>(dimension,0);
    
    random_w = vector<int>(dimension);
    
    for(int i = 0 ; i < dimension ; i++ )
        random_w[i] = i;
    random_shuffle(random_w.begin(),random_w.end());
    for(int i=0; i<dimension; i++)
    {
        for(uint j=0; j<vert; j++)
        {
            if(feat(j,i)>0)
                rowsum_pos[i]+=feat(j,i);
            else
                rowsum_neg[i]+=feat(j,i);
        }
    }
    
    vector<queue<uint>> candidate_sets(dimension);
    vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));
    if(algorithm == "instant"){
        cout<<"before push feat(357,37):"<<feat(357,37)<<endl;
        Instantgnn::ppr_push(dimension, feat, true,candidate_sets,isCandidates,true,algorithm, false);
        cout<<"after push feat(357,37):"<<feat(357,37)<<endl;
        
    }
    else if(algorithm == "speed_push"){
        cout<<"before push feat(357,37):"<<feat(357,37)<<endl;
        Instantgnn::ppr_push(dimension, feat, true,candidate_sets,isCandidates,true,algorithm, false);
        cout<<"after push feat(357,37):"<<feat(357,37)<<endl;
        cout<<"speed push"<<endl;
    }
  
    double dataset_size=(double)(((long long)edges+vert)*4+(long long)vert*dimension*8)/1024.0/1024.0/1024.0;
    return dataset_size;
}

void Instantgnn::ppr_push(int dimension, Eigen::Ref<Eigen::MatrixXd>feat, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, bool log, string algorithm, bool reverse)
{
    vector<thread> threads;
    
    struct timeval t_start,t_end;
    double timeCost;
    //clock_t start_t, end_t;
    gettimeofday(&t_start,NULL);
    if(log)
        cout<<"Begin propagation..."<<init << "...dimension:"<< dimension <<endl;
        cout<<"candidate_sets:"<<candidate_sets[1].front()<<endl;
    int ti,start;
    int ends=0;
    
    //start_t = clock();
    for( ti=1 ; ti <= dimension%NUMTHREAD ; ti++ )
    {
        start = ends;
        ends+=ceil((double)dimension/NUMTHREAD);
        if(init)
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,true,std::ref(candidate_sets),std::ref(isCandidates),algorithm,reverse));
        else
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,false,std::ref(candidate_sets),std::ref(isCandidates),algorithm, reverse));
    }
    for( ; ti<=NUMTHREAD ; ti++ )
    {
        start = ends;
        ends+=dimension/NUMTHREAD;
        if(init)
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,true,std::ref(candidate_sets),std::ref(isCandidates),algorithm,reverse));
        else
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,false,std::ref(candidate_sets),std::ref(isCandidates),algorithm,reverse));
    }
    
    for (int t = 0; t < NUMTHREAD ; t++)
        threads[t].join();
    vector<thread>().swap(threads);
    update_w.clear();
    
    //end_t = clock();
    //double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    gettimeofday(&t_end, NULL);
    timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
    if(log){
        cout<<"The propagation time: "<<timeCost<<" s"<<endl;
        //cout<<"The clock time : "<<total_t<<" s"<<endl;
    }
    string filename = "./time_accuracy.log";
    ofstream queryfile(filename, ios::app);
    queryfile<<"The propagation time when rmax = "<<rmax<<" is "<<timeCost<<endl;
    queryfile.close();
    vector<vector<bool>>().swap(isCandidates);
    vector<queue<uint>>().swap(candidate_sets);
}

// void Instantgnn::ppr_residue(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, string algorithm)
// {
//     // string algorithm = "instant";
//     // string algorithm = "speed_push";
//     if(algorithm=="instant"){
//         int w;
//         for(int it=st;it<ed;it++)
//         {
//             if(init)
//                 w = random_w[it];
//             else
//                 w = update_w[it];

            

//             queue<uint> candidate_set = candidate_sets[w];
//             vector<bool> isCandidate = isCandidates[w];

//             double rowsum_p=rowsum_pos[w];
//             double rowsum_n=rowsum_neg[w];
//             double rmax_p=rowsum_p*rmax;
//             double rmax_n=rowsum_n*rmax;// not same as the paper
//             if(rmax_n == 0) rmax_n = -rowsum_p;  

//             if(init)
//             {
//                 for(uint i=0; i<vert; i++)
//                 {
//                     R[w][i] = feats(i, w);
//                     feats(i, w) = 0;
                        
//                     if(R[w][i]>rmax_p || R[w][i]<rmax_n)
//                     {
//                         candidate_set.push(i);
//                         isCandidate[i] = true;
//                     }
//                 }
//             }

            
            
//             // for(uint i=0; i<vert; i++)// push from scratch
//             // {
//             //     R[w][i] = X(i, w);
//             //     feats(i, w) = 0;
                    
//             //     if(R[w][i]>rmax_p || R[w][i]<rmax_n)
//             //     {
//             //         candidate_set.push(i);
//             //         isCandidate[i] = true;
//             //     }
//             // }
            
//             // cout<<"initial candidate_set.size(): "<<candidate_set.size()<<endl;
//             int num = 0;

//             while(candidate_set.size() > 0)
//             {   
//                 num++;
//                 // if(num%5000==0){
//                 //     MSG(candidate_set.size());
//                 // }        
//                 uint tempNode = candidate_set.front();
//                 candidate_set.pop();
//                 isCandidate[tempNode] = false;
//                 double old = R[w][tempNode];
                
//                 R[w][tempNode] = 0;
//                 feats(tempNode,w) += alpha*old;
                
//                 uint inSize = g.getInSize(tempNode);
//                 for(uint i=0; i<inSize; i++)
//                 {
//                     uint v = g.getInVert(tempNode, i);
//                     R[w][v] += (1-alpha) * old / Du[v] / Du[tempNode];
//                     if(!isCandidate[v])
//                     {
//                         if(R[w][v] > rmax_p || R[w][v] < rmax_n)
//                         {
//                             candidate_set.push(v);
//                             isCandidate[v] = true;
//                         }
//                     }
//                 }
//             }

//             // //Random walk!
//             // unsigned long long num_random_walk = omega;
//             // double row_sum_n = 0,row_sum_p = 0;
//             // for(uint j=0; j<vert; j++)
//             // {
//             //     if(R[w][j]>0)
//             //         row_sum_p+=R[w][j];
//             //     else
//             //         row_sum_n+=R[w][j];
//             // }
//             // for(uint id=0; id < vert; id++){
//             //     double check_sum;
//             //     double residual = R[w][id];
//             //     if(residual<0){
//             //        check_sum = row_sum_n;
//             //     }else if(residual>0){
//             //        check_sum = row_sum_p;
//             //     }else{
//             //         continue;
//             //     }
//             //     unsigned long num_s_rw = ceil(residual/check_sum*num_random_walk);
//             //     double ppr_incre = check_sum/num_random_walk*Du[id];
//             //     num_total_rw += num_s_rw;
// 			// 	rw_count += num_s_rw;
//             //     for(unsigned long j=0; j<num_s_rw; j++){
//             //         int des = g.random_walk(id);
//             //         feats(des,w) += ppr_incre;
//             //     }
//             // }


//             // cout<<"1111111"<<endl;
//             vector<bool>().swap(isCandidates[w]);
//         }
//     }
//     else if(algorithm == "speed_push"){
//         SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(vert);
//         std::vector<PageRankScoreType> seed;
//         // cout<<"feat.rows():"<<feats.rows()<<" feat.cols():"<<feats.cols()<<"feats.col(i):"<<feats.col(54).size()<<endl;
//         // const type_info &objInfo = typeid(feats.col(54));
//         // cout<<"type::"<<objInfo.name()<<endl;
        
//         for (VertexIdType i = st; i < ed; i++) {
//             // push_one(i, graph_structure, seed);
            
            
//             // propagate_vector(feature_matrix[i], seed, vert, true);
            
           

//             ppr.calc_ppr_walk(graph_structure, feats, i, epsilon, alpha, lower_threshold);
      
//             // Save embedding vector of feature i on all nodes to out_matrix
         
//             // std::swap_ranges(_graph_structure.means.begin(), _graph_structure.means.end()-2,
//             //                 out_matrix[i%spt_size].begin());

//         }
//     }
// }


void Instantgnn::ppr_residue(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, string algorithm, bool reverse)
{
    // string algorithm = "instant";
    // string algorithm = "speed_push";
    if(algorithm=="instant"){
        int w;
        for(int it=st;it<ed;it++)
        {
            if(init)
                w = random_w[it];
            else
                w = update_w[it];

            
            vector<bool> push_round(4,false); 
            queue<uint> candidate_set = candidate_sets[w];
            vector<bool> isCandidate = isCandidates[w];

            double rowsum_p=rowsum_pos[w];
            double rowsum_n=rowsum_neg[w];
            double rmax_p=rowsum_p*rmax;
            double rmax_n=rowsum_n*rmax;// not same as the paper
            config.rbmax_p = rmax_p;
            config.rbmax_n = rmax_n;
            if(rmax_n == 0) rmax_n = -rowsum_p;  


            if(reverse){
                while(candidate_set.size() > 0)
                {                    
                    uint tempNode = candidate_set.front();
                    candidate_set.pop();
                    isCandidate[tempNode] = false;
                    double old = R_b[w][tempNode];
                    R_b[w][tempNode] = 0;
                    pi_b[w][tempNode] += alpha*old;
                    
                    uint inSize = g.getInSize(tempNode);
                    for(uint i=0; i<inSize; i++)
                    {
                        uint v = g.getInVert(tempNode, i);
                        R_b[w][v] += (1-alpha) * old / Du[v] / Du[tempNode];
                        if(!isCandidate[v])
                        {
                            if(R_b[w][v] > config.rbmax_p || R_b[w][v] < config.rbmax_n)
                            {
                                candidate_set.push(v);
                                isCandidate[v] = true;
                            }
                        }
                    }
                }
            }
            else{

                // if(init)
                // {
                //     for(uint i=0; i<vert; i++)
                //     {
                //         R[w][i] = feats(i, w);
                //         feats(i, w) = 0;
                //         if(R[w][i]>rmax_p || R[w][i]<rmax_n)
                //         {
                //             candidate_set.push(i);
                //             isCandidate[i] = true;
                //         }
                //     }
                // }

                // while(candidate_set.size() > 0)
                // {
                //     uint tempNode = candidate_set.front();
                //     candidate_set.pop();
                //     isCandidate[tempNode] = false;
                //     double old = R[w][tempNode];
                //     R[w][tempNode] = 0;
                //     feats(tempNode,w) += alpha*old;
                    
                //     uint inSize = g.getInSize(tempNode);
                //     for(uint i=0; i<inSize; i++)
                //     {
                //         uint v = g.getInVert(tempNode, i);
                //         R[w][v] += (1-alpha) * old / Du[v] / Du[tempNode];
                //         if(!isCandidate[v])
                //         {
                //             if(R[w][v] > rmax_p || R[w][v] < rmax_n)
                //             {
                //                 candidate_set.push(v);
                //                 isCandidate[v] = true;
                //             }
                //         }
                //     }
                // }
                //****my code
                if(init){
                    for(uint i=0; i<vert; i++){   
                        R[w][i] = feats(i, w);
                        feats(i, w) = 0;
                    }
                }
                // for(uint i=0; i<vert; i++){   //always push from scratch
                //     R[w][i] = X(i, w);
                //     feats(i, w) = 0;
                // }
                for(uint k = 0;k<4;k++){

                    for(uint i=0; i<vert; i++)
                    {   

                        if(R[w][i]>rmax_p || R[w][i]<rmax_n)
                        {
                            candidate_set.push(i);
                            isCandidate[i] = true;
                        }
                    }
                    // cout<<"initial candidate_set.size(): "<<candidate_set.size()<<endl;
                    int num = 0;
                    // MSG(candidate_set.size());
                    while(candidate_set.size() > 0)
                    {   
                        num++;
                        // if(num%5000==0){
                        //     MSG(candidate_set.size());
                        // }        
                        uint tempNode = candidate_set.front();
                        candidate_set.pop();
                        isCandidate[tempNode] = false;
                        double old = R[w][tempNode];
                        
                        R[w][tempNode] = 0;
                        feats(tempNode,w) += alpha*old;
                        
                        uint inSize = g.getInSize(tempNode);
                        for(uint i=0; i<inSize; i++)
                        {
                            uint v = g.getInVert(tempNode, i);
                            R[w][v] += (1-alpha) * old / Du[v] / Du[tempNode];
                        }
                    }
                }//*****my code
            }
            // //Random walk!
            // unsigned long long num_random_walk = omega;
            // double row_sum_n = 0,row_sum_p = 0;
            // for(uint j=0; j<vert; j++)
            // {
            //     if(R[w][j]>0)
            //         row_sum_p+=R[w][j];
            //     else
            //         row_sum_n+=R[w][j];
            // }
            // for(uint id=0; id < vert; id++){
            //     double check_sum;
            //     double residual = R[w][id];
            //     if(residual<0){
            //        check_sum = row_sum_n;
            //     }else if(residual>0){
            //        check_sum = row_sum_p;
            //     }else{
            //         continue;
            //     }
            //     unsigned long num_s_rw = ceil(residual/check_sum*num_random_walk);
            //     double ppr_incre = check_sum/num_random_walk*Du[id];
            //     num_total_rw += num_s_rw;
			// 	rw_count += num_s_rw;
            //     for(unsigned long j=0; j<num_s_rw; j++){
            //         int des = g.random_walk(id);
            //         feats(des,w) += ppr_incre;
            //     }
            // }


            // cout<<"1111111"<<endl;
            vector<bool>().swap(isCandidates[w]);
        }
    }
    else if(algorithm == "speed_push"){
        SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(vert);
        std::vector<PageRankScoreType> seed;
        // cout<<"feat.rows():"<<feats.rows()<<" feat.cols():"<<feats.cols()<<"feats.col(i):"<<feats.col(54).size()<<endl;
        // const type_info &objInfo = typeid(feats.col(54));
        // cout<<"type::"<<objInfo.name()<<endl;
        
        for (VertexIdType i = st; i < ed; i++) {
            // push_one(i, graph_structure, seed);
            
            
            // propagate_vector(feature_matrix[i], seed, vert, true);
            
           

            ppr.calc_ppr_walk(graph_structure, feats, i, epsilon, alpha, lower_threshold);
      
            // Save embedding vector of feature i on all nodes to out_matrix
         
            // std::swap_ranges(_graph_structure.means.begin(), _graph_structure.means.end()-2,
            //                 out_matrix[i%spt_size].begin());

        }
    }
}

}