#include "instantAlg.h"
#include "Graph.h"
#include "BatchRandomWalk.h"
#include "SpeedPPR.h"
#include "HelperFunctions.h"
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
            g.deleteEdge(v_from, v_to);
            delete_neighbors[v_from].push_back(v_to);
        }
    }
    infile.close();
    cout<<"update graph finish..."<<"affected_nodelst.size():"<<affected_nodelst.size()<<endl;

    return new_neighbors;
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
            feat(dim,affected_node) = feat(dim,affected_node) * Du[affected_node] / oldDu[i];
            double delta_1 = feat(dim,affected_node) * (oldDu[i]-Du[affected_node]) / alpha / Du[affected_node];
            R[dim][affected_node] += delta_1;
        }
    }
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
            
            double increment = feat(dim,affected_node) + alpha*R[dim][affected_node] - alpha*X(dim,affected_node);
            increment *= oldDu[i] - Du[affected_node];
            increment /= Du[affected_node];
            
            for(uint j=0; j<add_neighbors[affected_node].size(); j++)
            {
                uint add_node = add_neighbors[affected_node][j];
                increment += (1-alpha)*feat(dim,add_node) / Du[affected_node] / Du[add_node];
            }
            for(uint j=0; j<delete_neighbors[affected_node].size(); j++)
            {
                uint delete_node = delete_neighbors[affected_node][j];
                increment -= (1-alpha)*feat(dim,delete_node) / Du[affected_node] / Du[delete_node];
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
      Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm);
    }
}

void Instantgnn::linenum_operation(string updatefilename, int begin, int end, double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat, string algorithm)
{
    alpha=alphaa;
    rmax=rmaxx;
    
    int insertFLAG = 0;
    ifstream infile(updatefilename.c_str());
    int k = 0;
    uint v_from, v_to;
    cout<<"updating begin, for snapshot: " << updatefilename <<endl;
    
    string outputFile = "deltaZ.txt";
    
    int line_num = 0;
    while (infile >> v_from >> v_to)
    {
        line_num += 1;
        if(line_num <= begin)
            continue;
        else if (line_num > end)
            break;
        //cout << "line:" << line_num << " from:" << v_from << " to: " << v_to << endl;
        insertFLAG = g.isEdgeExist(v_from, v_to);
        //cout << "insertFLAG: " << insertFLAG << endl;
        // update graph
        if(insertFLAG == 1)
            g.insertEdge(v_from, v_to);
        else if(insertFLAG == -1)
            g.deleteEdge(v_from, v_to);
        // update Du
        double oldDu = Du[v_from];
        Du[v_from] = pow(g.getOutSize(v_from), 0.5);
        
        vector<queue<uint>> candidate_sets(dimension);
        vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));
        vector<bool> isUpdateW(dimension, false);
        
        for(int i=0; i<dimension; i++)
        {
            double rowsum_p=rowsum_pos[i];
            double rowsum_n=rowsum_neg[i];
            double rmax_p=rowsum_p*rmax;
            double rmax_n=rowsum_n*rmax;
            if(rmax_n == 0) rmax_n = -rmax_p; 
            double increment = feat(i,v_from) + alpha*R[i][v_from] - alpha*X(i,v_from);
            increment *= oldDu - Du[v_from];
            increment /= Du[v_from];
            double in_v = (1-alpha)*feat(i,v_to) / Du[v_from] / Du[v_to];
 
            if(insertFLAG > 0)
            {
                increment += in_v;
                increment /= alpha;
                R[i][v_from] += increment;
            }
            else  //delete edge
            {
                increment -= in_v;
                increment /= alpha;
                R[i][v_from] += increment;
            }
            if( R[i][v_from]>rmax_p || R[i][v_from]<rmax_n )
            {
                k++;
                if(!isCandidates[i][v_from]){
                    candidate_sets[i].push(v_from);
                    isCandidates[i][v_from] = true;
                }
                if(!isUpdateW[i]){
                    update_w.push_back(i);
                    isUpdateW[i] = true;
                }
            }
            for(uint j=0; j<g.getInSize(v_from); j++)
            {
                uint node_w = g.getInVert(v_from, j);
                double increment_w = (1-alpha) * feat(i,v_from) / Du[node_w];
                increment_w *= 1/Du[v_from] - 1/oldDu;
                increment_w /= alpha;
                
                R[i][node_w] += increment_w;
                //cout << increment_w<<endl;
                if( R[i][node_w]>rmax_p || R[i][node_w]<rmax_n )
                {
                    if(!isCandidates[i][node_w]){
                        candidate_sets[i].push(node_w);
                        isCandidates[i][node_w] = true;
                        k++;
                    }
                    if(!isUpdateW[i]){
                        update_w.push_back(i);
                        isUpdateW[i] = true;
                    }
                }
            } 
        }
        //push every edge
        if(update_w.size()>0)
            Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,false,algorithm);
    }
    infile.close();
}

int Instantgnn::snapshot_operation_rate_Z(string updatefilename, int begin, double rmaxx,double alphaa, double threshold, Eigen::Map<Eigen::MatrixXd> &feat, Eigen::Map<Eigen::MatrixXd> &init_Z, string algorithm)
{
    alpha=alphaa;
    rmax=rmaxx;

    int insertFLAG = 0;
    ifstream infile(updatefilename.c_str());
    int k = 0;
    int v_from, v_to;
    clock_t start_t, end_t;
    start_t = clock();
    cout<<"updating begin, for snapshot: " << updatefilename <<endl;
    
    string outputFile = updatefilename + "_rate_Z.txt";
    ofstream fout;
    fout.open(outputFile);
    fout.setf(ios::fixed,ios::floatfield);
    fout.precision(15);
    if(!fout) cout<<"Fail to open the writed file"<<endl;
    
    int line_num = 0;
    double rate=0;
    Eigen::MatrixXd feat_last_time, diff, delta_feat;
    feat_last_time = feat;
    while (infile >> v_from >> v_to)
    {
        line_num += 1;
        if(line_num <= begin)
            continue;
        
        //cout << "from:" << v_from << " to: " << v_to << endl;
        insertFLAG = g.isEdgeExist(v_from, v_to);
        //cout << "insertFLAG: " << insertFLAG << endl;
        // update graph
        if(insertFLAG == 1)
            g.insertEdge(v_from, v_to);
        else if(insertFLAG == -1)
            g.deleteEdge(v_from, v_to);
        // update Du
        double oldDu = Du[v_from];
        Du[v_from] = pow(g.getOutSize(v_from), 0.5);
        
        vector<queue<uint>> candidate_sets(dimension);
        vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));
        vector<bool> isUpdateW(dimension, false);
        
        for(int i=0; i<dimension; i++)
        {
            double rowsum_p=rowsum_pos[i];
            double rowsum_n=rowsum_neg[i];
            double rmax_p=rowsum_p*rmax;
            double rmax_n=rowsum_n*rmax;
            if(rmax_n == 0) rmax_n = -rmax_p; 
            double increment = feat(i,v_from) + alpha*R[i][v_from] - alpha*X(i,v_from);
            increment *= oldDu - Du[v_from];
            increment /= Du[v_from];
            double in_v = (1-alpha)*feat(i,v_to) / Du[v_from] / Du[v_to];
 
            if(insertFLAG > 0)
            {
                increment += in_v;
                increment /= alpha;
                R[i][v_from] += increment;
            }
            else  //delete edge
            {
                increment -= in_v;
                increment /= alpha;
                R[i][v_from] += increment;
            }
            if( R[i][v_from]>rmax_p || R[i][v_from]<rmax_n )
            {
                k++;
                if(!isCandidates[i][v_from]){
                    candidate_sets[i].push(v_from);
                    isCandidates[i][v_from] = true;
                }
                if(!isUpdateW[i]){
                    update_w.push_back(i);
                    isUpdateW[i] = true;
                }
            }
            for(uint j=0; j<g.getInSize(v_from); j++)
            {
                uint node_w = g.getInVert(v_from, j);
                double increment_w = (1-alpha) * feat(v_from,i) / Du[node_w];
                increment_w *= 1/Du[v_from] - 1/oldDu;
                increment_w /= alpha;
                
                R[i][node_w] += increment_w;
                //cout << increment_w<<endl;
                if( R[i][node_w]>rmax_p || R[i][node_w]<rmax_n )
                {
                    if(!isCandidates[i][node_w]){
                        candidate_sets[i].push(node_w);
                        isCandidates[i][node_w] = true;
                        k++;
                    }
                    if(!isUpdateW[i]){
                        update_w.push_back(i);
                        isUpdateW[i] = true;
                    }
                }
            } 
        }
        //push every edge and cal diff every edge
        //cout<<"k = "<<k <<", update_w.size:"<<update_w.size()<<endl;
        if(update_w.size()>0)
        {
          Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,false,algorithm);
          diff = feat - feat_last_time;
          delta_feat = feat - init_Z;
          double diff_F = diff.norm();
          double delta_feat_F = delta_feat.norm();
          rate = diff_F / delta_feat_F;
          //double diff_l1 = diff.lpNorm<1>();
          fout<<line_num<<" "<<diff_F<<" "<< delta_feat_F << " " << rate <<endl;
        }else{
          fout<<line_num<<" "<<0<<" "<< 0 << " " << 0 <<endl;
        }
        if(rate>threshold)
            break;
    }
    infile.close();
    fout.close();
    end_t = clock();
    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    cout<<"snapshot time: " << total_t << endl;
    return line_num;
}

void Instantgnn::overall_operation(double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat, string algorithm){
    alpha=alphaa;
    rmax=rmaxx;
    
    vector<queue<uint>> candidate_sets(dimension);
    vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));
    vector<bool> isUpdateW(dimension, false);
    for(int i=0; i<dimension; i++)
    {
        double rowsum_p=rowsum_pos[i];
        double rowsum_n=rowsum_neg[i];
        double rmax_p=rowsum_p*rmax;
        double rmax_n=rowsum_n*rmax;
        if(rmax_n == 0) rmax_n = -rmax_p;
        for(uint j=0; j<vert; j++)
        {
            if( R[i][j]>rmax_p || R[i][j]<rmax_n )
            {
                if(!isCandidates[i][j]){
                    candidate_sets[i].push(j);
                    isCandidates[i][j] = true;
                }
                if(!isUpdateW[i]){
                    update_w.push_back(i);
                    isUpdateW[i] = true;
                }
            }
        }
    }
    if(update_w.size()>0)
    {
        Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm);
    }
}

int startsWith(string s, string sub){
        return s.find(sub)==0?1:0;
}
double Instantgnn::initial_operation(string path, string dataset,uint mm,uint nn,double rmaxx,double alphaa,double epsilonn,Eigen::Map<Eigen::MatrixXd> &feat, string algorithm)
{
    X = feat; // change in feat not influence X
    /*if(startsWith(dataset, "arxiv")){
        cout<<"X.cols():"<<X.cols()<<endl;
        X = X.transpose().eval();
        cout<<"X.cols():"<<X.cols()<<endl;
    }*/
    rmax=rmaxx;
    edges=mm;
    vert=nn;
    alpha=alphaa;
    dataset_name=dataset;
    cout<<dataset_name<<endl;
    g.inputGraph(path, dataset_name, vert, edges);
    cout<<"g.getOutSize:"<<g.getOutSize(0)<<endl;

    dimension=min(feat.rows(),feat.cols());
    cout<<"dimension: "<<dimension<<", col:"<<max(feat.rows(),feat.cols())<<endl;
    cout<<" 1..."<<endl;
    Du=vector<double>(vert,0);
    double rrr=0.5;
    for(uint i=0; i<vert; i++)
    {
        Du[i]=pow(g.getOutSize(i),rrr);
    }

    R = vector<vector<double>>(dimension, vector<double>(vert, 0));
    cout<<" 3..."<<endl;
    rowsum_pos = vector<double>(dimension,0);
    cout<<" 3..."<<endl;
    rowsum_neg = vector<double>(dimension,0);
    cout<<" 3..."<<endl;
    random_w = vector<int>(dimension);
    
    for(int i = 0 ; i < dimension ; i++ )
        random_w[i] = i;
    
    random_shuffle(random_w.begin(),random_w.end());
    for(int i=0; i<dimension; i++)
    {
        for(uint j=0; j<vert; j++)
        {
            if(feat(i,j)>0)
                rowsum_pos[i]+=feat(i,j);
            else
                rowsum_neg[i]+=feat(i,j);
        }
    }
    cout<<" 4..."<<endl;
    vector<queue<uint>> candidate_sets(dimension);
    vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));
    cout<<" Start Push..."<<endl;
    Instantgnn::ppr_push(dimension, feat, true,candidate_sets,isCandidates,true,algorithm);

    double dataset_size=(double)(((long long)edges+vert)*4+(long long)vert*dimension*8)/1024.0/1024.0/1024.0;
    return dataset_size;
}

void Instantgnn::ppr_push(int dimension, Eigen::Ref<Eigen::MatrixXd>feat, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, bool log, string algorithm)
{
    vector<thread> threads;
    
    struct timeval t_start,t_end;
    double timeCost;
    //clock_t start_t, end_t;
    gettimeofday(&t_start,NULL);
    if(log)
        cout<<"Begin propagation..."<<init << "...dimension:"<< dimension <<endl;
    int ti,start;
    int ends=0;
    
    //start_t = clock();
    for( ti=1 ; ti <= dimension%NUMTHREAD ; ti++ )
    {
        start = ends;
        ends+=ceil((double)dimension/NUMTHREAD);
        if(init)
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,true,std::ref(candidate_sets),std::ref(isCandidates),algorithm));
        else
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,false,std::ref(candidate_sets),std::ref(isCandidates),algorithm));
    }
    for( ; ti<=NUMTHREAD ; ti++ )
    {
        start = ends;
        ends+=dimension/NUMTHREAD;
        if(init)
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,true,std::ref(candidate_sets),std::ref(isCandidates),algorithm));
        else
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,false,std::ref(candidate_sets),std::ref(isCandidates),algorithm));
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
    vector<vector<bool>>().swap(isCandidates);
    vector<queue<uint>>().swap(candidate_sets);
}

void Instantgnn::ppr_residue(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, string algorithm)
{
    int w;
    for(int it=st;it<ed;it++)
    {
        if(init)
            w = random_w[it];
        else
            w = update_w[it];
            
        queue<uint> candidate_set = candidate_sets[w];
        vector<bool> isCandidate = isCandidates[w];

        double rowsum_p=rowsum_pos[w];
        double rowsum_n=rowsum_neg[w];
        double rmax_p=rowsum_p*rmax;
        double rmax_n=rowsum_n*rmax;
        if(rmax_n == 0) rmax_n = -rowsum_p;  

        if(init)
        {
            for(uint i=0; i<vert; i++)
            {
                R[w][i] = feats(w, i);
                feats(w, i) = 0;
                if(R[w][i]>rmax_p || R[w][i]<rmax_n)
                {
                     candidate_set.push(i);
                     isCandidate[i] = true;
                }
            }
        }

        while(candidate_set.size() > 0)
        {
            uint tempNode = candidate_set.front();
            candidate_set.pop();
            isCandidate[tempNode] = false;
            double old = R[w][tempNode];
            R[w][tempNode] = 0;
            feats(w,tempNode) += alpha*old;
            
            uint inSize = g.getInSize(tempNode);
            for(uint i=0; i<inSize; i++)
            {
                uint v = g.getInVert(tempNode, i);
                R[w][v] += (1-alpha) * old / Du[v] / Du[tempNode];
                if(!isCandidate[v])
                {
                    if(R[w][v] > rmax_p || R[w][v] < rmax_n)
                    {
                        candidate_set.push(v);
                        isCandidate[v] = true;
                    }
                }
            }
        }
        vector<bool>().swap(isCandidates[w]);
    }
}

}